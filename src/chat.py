import json
import re
from pathlib import Path

import faiss
import numpy as np
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
        self.catalog_path = Path(__file__).resolve().parents[1] / "data" / "mit_courses.json"

        self.system_prompt = (
            "You are an MIT Course Catalog assistant helping students choose classes. "
            "Recommend a short list of relevant classes and explain why each fits the student's constraints. "
            "When the user asks for recommendations, you should ask concise follow-up questions to clarify constraints: major/course, class year,provide concrete course suggestions immediately. "
            "requirements (CI-H, HASS, REST, GIRs), schedule preferences, and interests. "
            "Don't over do it with the follow-up questions.Once you have the constraints, provide concrete course suggestions immediately."
            "Do not repeatedly ask for the same information if the user already provided it earlier in the conversation. "
            "If schedule/instructor details might vary by term, clearly say they should verify in the live catalog "
            "at student.mit.edu/catalog before enrolling. "
            "Always prioritize the user's latest question over earlier turns. "
            "Be practical, organized, and concise. "
            "Do not invent classes. Prioritize the provided context; if context is limited, say so clearly."
            "If nothing useful in the context comes up, you can go and look for yourself on student.mit.edu/catalog before enrolling."
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
                f"description: {(c.get('raw_text') or '')[:280]}"
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

    def _retrieve_top_chunks(self, user_input, top_k=4):
        """
        Query-time retrieval from FAISS plus exact course-code matches.
        """
        if self.index is None or not self.chunks:
            return []

        selected = []
        selected_ids = set()

        # Always include exact course references when user names one.
        for code in self._extract_course_codes(user_input):
            chunk = self.chunk_by_id.get(code.lower())
            if chunk and chunk["id"] not in selected_ids:
                selected.append(
                    {
                        "id": chunk["id"],
                        "score": 1.0,
                        "text": chunk["text"],
                        "source": "exact",
                    }
                )
                selected_ids.add(chunk["id"])
                if len(selected) >= top_k:
                    return selected

        q = self.embedder.encode(
            [user_input],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        # Pull extra neighbors so we can dedupe around exact matches.
        scores, indices = self.index.search(q, top_k + 6)
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

    def _build_retrieval_context(self, user_input):
        """
        Build prompt context from retrieved chunks.
        """
        hits = self._retrieve_top_chunks(user_input, top_k=6)
        if not hits:
            return "No matching course chunks were found in the local MIT dataset."

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
                    "Retrieved context:\n"
                    f"{retrieval_context}\n\n"
                    "Use this context first when answering.\n"
                    "Behavior rules:\n"
                    "- Do not ask for constraints already provided by the user in history.\n"
                    "- For recommendation requests, return 3-5 concrete courses first.\n"
                    "- Keep follow-up questions optional and brief."
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
                if role in {"user", "assistant"} and isinstance(content, str) and content:
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
        retrieval_context = self._build_retrieval_context(user_input)

        try:
            completion = self.client.chat_completion(
                messages=self._build_messages(user_input, history, retrieval_context),
                max_tokens=450,
                temperature=0.2,
            )
            text = self._extract_chat_text(completion)
            if text:
                return text
        except Exception as exc:
            print(f"[chat_completion warning] {exc}")

        return "I am having trouble generating a response right now. Please try again in a few seconds."
