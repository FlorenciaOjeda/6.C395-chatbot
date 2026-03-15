import json
import re
from pathlib import Path

import faiss
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from config import BASE_MODEL, MY_MODEL, HF_TOKEN


class Chatbot:
    """
    Local RAG chatbot using:
    1) one chunk per course
    2) embeddings
    3) FAISS similarity search
    """

    def __init__(self):
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.catalog_path = (
            Path(__file__).resolve().parents[1] / "data" / "mit_courses.json"
        )

        self.system_prompt = (
            "You are an MIT Course Catalog assistant helping students choose classes. "
            "YOUR PRIMARY JOB IS TO RECOMMEND COURSES. Always lead with 3-5 concrete course recommendations. "
            "RULES — follow these strictly:\n"
            "1. ALWAYS give course recommendations first, even if you have only partial information. "
            "Use whatever constraints the user has given (major, interests, schedule, requirements) and recommend immediately.\n"
            "2. Never ask more than ONE follow-up question, and only ask if a critical constraint is completely missing. "
            "If the user has mentioned a course number, interests, or schedule preferences, that is enough — recommend now.\n"
            "3. Never list multiple questions. If you must ask, pick the single most important one and ask it after your recommendations.\n"
            "4. Do not ask for information already provided earlier in the conversation.\n"
            "5. Do not invent courses. Use the retrieved context; if context is limited, say so clearly.\n"
            "6. Schedule and instructor details can vary by term — always tell the user to verify at student.mit.edu/catalog before enrolling.\n"
            "7. Be concise and well-organized. Use bullet points or numbered lists for course recommendations.\n"
            "8. After your recommendations, add a 'References' section listing each recommended course and its catalog URL "
            "from the retrieved context (the 'url:' field). Format each line as: '- Course Number Title: <url>'. "
            "Do not include URLs inline within the recommendations themselves."
        )

        self.chunks = self._load_chunks()
        self.chunk_by_id = {c["id"].lower(): c for c in self.chunks if c.get("id")}
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index = self._build_faiss_index(self.chunks)

    def _load_chunks(self):
        """
        Prepare data: one text chunk per course.
        """
        try:
            payload = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            courses = payload.get("courses", [])
        except Exception:
            courses = []

        chunks = []
        for c in courses:
            schedule_lines = c.get("schedule_lines", [])
            schedule_text = " | ".join(schedule_lines[:2]) if schedule_lines else ""
            chunk_text = (
                f"course: {c.get('number', '')} {c.get('title', '')}\n"
                f"prereq: {c.get('prerequisites', '')}\n"
                f"units: {c.get('units', '')}\n"
                f"tags: {', '.join(c.get('requirement_tags', []))}\n"
                f"schedule: {schedule_text}\n"
                f"url: {c.get('source_page', '')}#{c.get('number', '')}\n"
                f"description: {(c.get('raw_text') or '')[:600]}"
            )
            chunks.append(
                {
                    "id": c.get("number", ""),
                    "text": chunk_text,
                }
            )
        return chunks

    def _build_faiss_index(self, chunks):
        """
        Generate embeddings and store in FAISS.
        """
        if not chunks:
            return None

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine when vectors are normalized
        index.add(embeddings)
        return index

    def _extract_course_codes(self, text):
        """
        Find explicit course codes like 24.131, 6.3900, STS.085.
        """
        pattern = r"\b(?:[A-Za-z]{2,5}\.)?\d+[A-Za-z]?(?:\.\d+[A-Za-z\[\]]*)?\b"
        raw_codes = re.findall(pattern, text or "")
        codes = []
        for code in raw_codes:
            cleaned = code.strip().strip(".,:;()").upper()
            if "." in cleaned and len(cleaned) >= 4:
                codes.append(cleaned)
        return codes

    def _extract_schedule_constraints(self, text):
        """
        Parse natural language schedule constraints.
        Returns (exclude_days, exclude_times) where:
          exclude_days  — set of MIT day letters to block (M T W R F)
          exclude_times — list of time strings to block (e.g. ['8.30', '9'])
        """
        t = text.lower()
        neg = r"\b(no|avoid|without|not on)\b"

        exclude_days = set()
        day_map = {
            "M": ["monday", "mondays"],
            "T": ["tuesday", "tuesdays"],
            "W": ["wednesday", "wednesdays"],
            "R": ["thursday", "thursdays"],
            "F": ["friday", "fridays"],
        }
        for code, keywords in day_map.items():
            for kw in keywords:
                if re.search(rf"{neg}[^.!?\n]{{0,25}}\b{kw}\b", t):
                    exclude_days.add(code)

        exclude_times = []

        # Explicit AM times: "no 8:30", "no 8:30am", "no 9am"
        for m in re.finditer(
            r"\b(no|avoid)\b[^.!?\n]{0,20}?(\d{1,2})[:\.]?(\d{2})?\s*am\b", t
        ):
            hour = int(m.group(2))
            minute = m.group(3) or "00"
            if 7 <= hour <= 11:
                exclude_times.append(
                    f"{hour}.{minute}" if minute != "00" else str(hour)
                )

        # Explicit PM times: "no 4pm", "no 4:30pm", "no 5pm"
        # MIT schedule uses 1-9 for 1pm-9pm directly (no am/pm suffix)
        for m in re.finditer(
            r"\b(no|avoid)\b[^.!?\n]{0,20}?(\d{1,2})[:\.]?(\d{2})?\s*pm\b", t
        ):
            hour = int(m.group(2))
            minute = m.group(3) or "00"
            if 1 <= hour <= 9:
                exclude_times.append(
                    f"{hour}.{minute}" if minute != "00" else str(hour)
                )

        # Bare time with no am/pm: "no 8:30", "absolutely no 830"
        for m in re.finditer(
            r"\b(no|avoid)\b[^.!?\n]{0,20}?(\d{1,2})[:\.](\d{2})(?!\s*(?:am|pm))", t
        ):
            hour = int(m.group(2))
            minute = m.group(3)
            if 7 <= hour <= 11:  # ambiguous — assume AM
                exclude_times.append(f"{hour}.{minute}")

        # Broad "early morning" shorthand
        if re.search(r"\b(no early|avoid early|early morning free|no morning)\b", t):
            for et in ["8", "8.30", "9"]:
                if et not in exclude_times:
                    exclude_times.append(et)

        # Broad "late afternoon" / "evening" shorthand
        if re.search(
            r"\b(no late afternoon|avoid late afternoon|late afternoon free)\b", t
        ):
            for et in ["4", "4.30", "5"]:
                if et not in exclude_times:
                    exclude_times.append(et)
        if re.search(r"\b(no evening|avoid evening|evening free)\b", t):
            for et in ["5", "5.30", "6", "7"]:
                if et not in exclude_times:
                    exclude_times.append(et)

        return exclude_days, exclude_times

    def _chunk_violates_schedule(self, chunk_text, exclude_days, exclude_times):
        """
        Return True if this chunk's schedule contains an excluded day or time.
        MIT schedule format: day-letters directly followed by a time, e.g. MW9.30, F2, TR11.
        """
        if not exclude_days and not exclude_times:
            return False

        m = re.search(r"schedule:\s*(.+?)(?:\n|$)", chunk_text)
        if not m:
            return False  # no schedule info — don't exclude
        schedule = m.group(1)

        # Extract day blocks: one or more day letters immediately before a digit
        day_blocks = re.findall(r"([MTWRF]+)(\d)", schedule)
        for days, _ in day_blocks:
            for day in exclude_days:
                if day in days:
                    return True

        # Check excluded times: day block immediately followed by the time string
        for t in exclude_times:
            if re.search(rf"[MTWRF]+{re.escape(t)}(?:[^0-9]|$)", schedule):
                return True

        return False

    def _retrieve_top_chunks(self, user_input, top_k=10):
        """
        Query-time retrieval from FAISS plus exact course-code matches.
        When a course code is mentioned, its chunk text is used as an additional
        seed query so FAISS finds similar courses, not just the named one.
        """
        if self.index is None or not self.chunks:
            return []

        selected = []
        selected_ids = set()
        seed_texts = []  # course chunk texts to use as extra seed queries

        # Always include exact course references when user names one,
        # and collect their text to seed similarity search.
        for code in self._extract_course_codes(user_input):
            chunk = self.chunk_by_id.get(code.lower())
            if chunk:
                seed_texts.append(chunk["text"])
                if chunk["id"] not in selected_ids:
                    selected.append(
                        {
                            "id": chunk["id"],
                            "score": 1.0,
                            "text": chunk["text"],
                            "source": "exact",
                        }
                    )
                    selected_ids.add(chunk["id"])

        # Build a combined query from the user message + any seed course texts
        # so that "find something like 6.C395" retrieves courses similar to 6.C395.
        combined_text = user_input
        if seed_texts:
            combined_text = user_input + " " + " ".join(seed_texts)

        q = self.embedder.encode(
            [combined_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Pull extra neighbors so we can dedupe around exact matches.
        scores, indices = self.index.search(q, top_k + len(selected_ids) + 6)
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk_id = self.chunks[idx]["id"]
            if chunk_id in selected_ids:
                continue
            selected.append(
                {
                    "id": chunk_id,
                    "score": float(score),
                    "text": self.chunks[idx]["text"],
                    "source": "semantic",
                }
            )
            selected_ids.add(chunk_id)
            if len(selected) >= top_k:
                break
        return selected

    def _history_query_context(self, history, max_turns=3):
        """
        Extract a plain-text summary of the most recent conversation turns to
        enrich the retrieval query. Keeps the last `max_turns` user+assistant pairs.
        """
        parts = []
        turns = []
        for item in history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content", "")
                if (
                    role == "user"
                    and isinstance(content, str)
                    and content
                ):
                    turns.append(content)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, _ = item
                if isinstance(user_msg, str) and user_msg:
                    turns.append(user_msg)

        # Take only the most recent turns to avoid diluting the query.
        for text in turns[-(max_turns * 2) :]:
            parts.append(text)
        return " ".join(parts)

    def _build_retrieval_context(self, user_input, history=None):
        """
        Build prompt context from retrieved chunks, enriched with conversation history.
        """
        # Build a combined query: recent history + current message so that
        # follow-up questions ("tell me more about #2", "less math?") still
        # retrieve relevant chunks.
        history_text = self._history_query_context(history or [])
        combined_query = (
            f"{history_text} {user_input}".strip() if history_text else user_input
        )

        # Soft-bias: if a department is mentioned, repeat it to weight the embedding
        dept_match = re.search(r"\bcourse\s+(\d+)\b", combined_query, re.IGNORECASE)
        if dept_match:
            dept_num = dept_match.group(1)
            combined_query = f"course {dept_num} {combined_query}"

        hits = self._retrieve_top_chunks(combined_query, top_k=10)
        if not hits:
            return "No matching course chunks were found in the local MIT dataset."

        # Filter out chunks that violate explicit schedule constraints.
        exclude_days, exclude_times = self._extract_schedule_constraints(combined_query)
        if exclude_days or exclude_times:
            filtered = [
                h
                for h in hits
                if not self._chunk_violates_schedule(
                    h["text"], exclude_days, exclude_times
                )
            ]
            # Only apply filter if it leaves enough results to be useful.
            if len(filtered) >= 3:
                hits = filtered

        lines = ["Top retrieved course chunks:"]
        for hit in hits:
            lines.append(
                f"[chunk {hit['id']}] source={hit.get('source', 'semantic')} similarity={hit['score']:.4f}"
            )
            lines.append(hit["text"])
        return "\n".join(lines)

    def _build_messages(self, user_input, history, retrieval_context):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "system",
                "content": (
                    "Retrieved context (use this as your primary source for recommendations):\n"
                    f"{retrieval_context}"
                ),
            },
        ]

        # Support both Gradio history formats: -> I did this because I got errors when only one was supported.
        # legacy list of [user_msg, assistant_msg]
        # message list of {"role": "...", "content": "..."}
        for item in history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if (
                    role in {"user", "assistant"}
                    and isinstance(content, str)
                    and content
                ):
                    messages.append({"role": role, "content": content})
                continue

            if isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, assistant_msg = item
                if isinstance(user_msg, str) and user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if isinstance(assistant_msg, str) and assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": user_input})
        return messages

    def _extract_chat_text(self, completion):
        try:
            content = completion.choices[0].message.content
        except Exception:
            return ""

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content") or ""
                    if isinstance(text, str) and text:
                        parts.append(text)
            return " ".join(parts).strip()
        if isinstance(content, dict):
            text = content.get("text") or content.get("content") or ""
            if isinstance(text, str):
                return text.strip()
        return ""

    def get_response(self, user_input, history=None):
        history = history or []
        retrieval_context = self._build_retrieval_context(user_input, history)

        try:
            completion = self.client.chat_completion(
                messages=self._build_messages(user_input, history, retrieval_context),
                max_tokens=650,
                temperature=0.2,
            )
            text = self._extract_chat_text(completion)
            if text:
                return text
        except Exception as exc:
            print(f"[chat_completion warning] {exc}")

        return "I am having trouble generating a response right now. Please try again in a few seconds."
