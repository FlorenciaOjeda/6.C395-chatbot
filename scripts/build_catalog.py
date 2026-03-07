#!/usr/bin/env python3
"""
Scrape MIT catalog pages and build a local JSON course dataset.

Usage:
    python scripts/build_catalog.py
    python scripts/build_catalog.py --output data/mit_courses.json --delay 0.15
    python scripts/build_catalog.py --max-pages 30   # useful for quick testing
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


BASE_INDEX_URL = "https://student.mit.edu/catalog/index.cgi"
CATALOG_HOST = "student.mit.edu"
CATALOG_PATH_PREFIX = "/catalog/"

USER_AGENT = (
    "Mozilla/5.0 (compatible; MITCatalogScraper/1.0; "
    "+https://student.mit.edu/catalog/index.cgi)"
)

NON_REQUIREMENT_ALTS = {
    "",
    "______",
    "Undergrad",
    "Graduate",
    "Fall",
    "IAP",
    "Spring",
    "Summer",
}

SECTION_RE = re.compile(r"<h3[^>]*>(.*?)</h3>(.*?)(?=<h3[^>]*>|</body>)", re.I | re.S)
HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.I)
IMG_ALT_RE = re.compile(r'<img[^>]+alt=["\']([^"\']*)["\']', re.I)


def fetch(url: str, timeout: int = 25, retries: int = 3, backoff_sec: float = 1.0) -> str:
    """Fetch an HTML page with basic retry behavior."""
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except (URLError, HTTPError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff_sec * attempt)
                continue
            break

    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def normalize_whitespace(text: str) -> str:
    """Normalize multi-line text while preserving line boundaries."""
    lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def strip_html(html: str) -> str:
    """Convert an HTML fragment into plain text."""
    cleaned = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    cleaned = re.sub(r"(?is)<style.*?>.*?</style>", " ", cleaned)
    cleaned = re.sub(r"(?i)<br\\s*/?>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</p\\s*>", "\n", cleaned)
    cleaned = re.sub(r"(?i)</div\\s*>", "\n", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    return normalize_whitespace(cleaned)


def extract_catalog_links(html: str, page_url: str) -> list[str]:
    """Extract same-site MIT catalog m*.html links from a page."""
    links: set[str] = set()
    for href in HREF_RE.findall(html):
        absolute = urljoin(page_url, href).split("#", 1)[0]
        parsed = urlparse(absolute)

        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc != CATALOG_HOST:
            continue
        if not parsed.path.startswith(CATALOG_PATH_PREFIX):
            continue

        # Keep only department/course pages (m*.html).
        filename = parsed.path.rsplit("/", 1)[-1]
        if not re.match(r"^m[0-9A-Za-z]+\.[Hh][Tt][Mm][Ll]$", filename):
            continue

        links.add(absolute)
    return sorted(links)


def parse_course_heading(heading_text: str) -> tuple[str | None, str | None]:
    """
    Parse the course number and title from an h3 heading.

    Examples:
      - "6.1000 Introduction to Programming..."
      - "24.C40[J] Ethics of Computing"
    """
    heading = re.sub(r"\s+", " ", heading_text).strip()
    match = re.match(r"^([A-Za-z0-9]+\.[A-Za-z0-9\[\]]+)\s*(.*)$", heading)
    if not match:
        return None, None

    number = match.group(1).strip()
    title = match.group(2).strip()
    return number, title


def parse_course_section(h3_html: str, section_html: str, page_url: str) -> dict[str, Any] | None:
    """Parse a single course section."""
    heading_text = strip_html(h3_html)
    number, title = parse_course_heading(heading_text)
    if not number:
        return None

    section_text = strip_html(section_html)
    if not section_text:
        return None

    prereq_match = re.search(r"(?:^|\n)Prereq:\s*(.+?)(?:\n|$)", section_text, flags=re.I)
    units_match = re.search(r"(?:^|\n)Units:\s*(.+?)(?:\n|$)", section_text, flags=re.I)

    schedule_prefixes = ("Lecture:", "Recitation:", "Lab:", "Studio:", "Seminar:", "Design:")
    schedule_lines: list[str] = []
    for line in section_text.splitlines():
        if line.startswith(schedule_prefixes):
            schedule_lines.append(line)

    requirement_tags: list[str] = []
    for alt in IMG_ALT_RE.findall(section_html):
        alt_clean = alt.strip()
        if alt_clean in NON_REQUIREMENT_ALTS:
            continue
        if alt_clean not in requirement_tags:
            requirement_tags.append(alt_clean)

    return {
        "number": number,
        "title": title,
        "heading": heading_text,
        "prerequisites": prereq_match.group(1).strip() if prereq_match else "",
        "units": units_match.group(1).strip() if units_match else "",
        "schedule_lines": schedule_lines,
        "requirement_tags": requirement_tags,
        "source_page": page_url,
        "raw_text": section_text,
    }


def parse_courses_from_page(html: str, page_url: str) -> list[dict[str, Any]]:
    """Extract all course sections from a catalog page."""
    courses: list[dict[str, Any]] = []
    for h3_html, section_html in SECTION_RE.findall(html):
        course = parse_course_section(h3_html, section_html, page_url)
        if course:
            courses.append(course)
    return courses


def crawl_catalog(delay_sec: float = 0.1, max_pages: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Crawl MIT catalog pages and return course records + crawl metadata."""
    index_html = fetch(BASE_INDEX_URL)
    seed_pages = extract_catalog_links(index_html, BASE_INDEX_URL)

    to_visit = deque(seed_pages)
    visited: set[str] = set()
    all_courses: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    while to_visit:
        if max_pages is not None and len(visited) >= max_pages:
            break

        page_url = to_visit.popleft()
        if page_url in visited:
            continue

        visited.add(page_url)
        try:
            html = fetch(page_url)
        except Exception as exc:  # noqa: BLE001 - keep scraping if one page fails
            errors.append({"url": page_url, "error": str(exc)})
            continue

        page_courses = parse_courses_from_page(html, page_url)
        all_courses.extend(page_courses)

        for link in extract_catalog_links(html, page_url):
            if link not in visited:
                to_visit.append(link)

        if len(visited) % 20 == 0:
            print(f"Visited {len(visited)} pages, collected {len(all_courses)} courses...")

        if delay_sec > 0:
            time.sleep(delay_sec)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "index_url": BASE_INDEX_URL,
        "pages_visited": len(visited),
        "seed_pages": len(seed_pages),
        "errors": errors,
    }
    return all_courses, metadata


def dedupe_courses(courses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe by (number, source_page) so cross-listed pages do not duplicate endlessly."""
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for course in courses:
        key = (course["number"], course["source_page"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(course)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MIT course dataset from student.mit.edu catalog")
    parser.add_argument(
        "--output",
        default="data/mit_courses.json",
        help="Path to output JSON file (default: data/mit_courses.json)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between requests (default: 0.1)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page cap for testing/debugging",
    )
    args = parser.parse_args()

    print("Starting MIT catalog crawl...")
    courses, metadata = crawl_catalog(delay_sec=args.delay, max_pages=args.max_pages)
    courses = dedupe_courses(courses)
    courses.sort(key=lambda c: (c["number"], c["title"]))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            **metadata,
            "course_count": len(courses),
        },
        "courses": courses,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Done. Wrote {len(courses)} courses to {output_path}")
    if metadata["errors"]:
        print(f"Encountered {len(metadata['errors'])} page fetch errors. See metadata.errors in output JSON.")


if __name__ == "__main__":
    main()
