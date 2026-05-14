"""Deterministic fake providers for DEMO_MODE.

When `DEMO_MODE=1` is set in the environment, the registry is bootstrapped
with these fakes instead of the real Anthropic / Tesseract / etc. Reviewers
can therefore click through the entire UI end-to-end without an API key
and without any binary dependencies.

The fakes are deterministic — same input always yields the same output.
That makes the trace inspector predictable and the demo reproducible.
"""
from __future__ import annotations
import hashlib
import json
import re
from providers.base import (
    EmbeddingProvider, LLMProvider, LLMResponse, OCRProvider, OCRResult,
    VectorStore, StoredChunk,
)


# ---------------------------------------------------------------------- LLM
class FakeLLM(LLMProvider):
    """Pattern-matches the system prompt to decide what to return.

    Covers the four call shapes the real system makes: structured-field
    extraction, planner queries, the case-fact-summary generator, the
    grounding judge, the edit signal classifier, and the simulated operator.
    """

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> LLMResponse:
        user = self._user_text(messages)

        if "structured fields" in system.lower():
            text = self._structured_fields(user)
        elif "retrieval queries" in system.lower():
            text = '{"queries": ["parties to the agreement", "effective date and payment terms"]}'
        elif "grounding judge" in system.lower():
            text = '{"supported": true, "reason": "evidence clearly states the claim"}'
        elif "extract a reusable rule" in system.lower():
            text = self._edit_signal(user)
        elif "senior associate" in system.lower():
            text = self._simulated_edit(user)
        elif "case fact summary" in system.lower() or "[chunk_id]" in system:
            text = self._case_fact_summary(user)
        else:
            text = "OK"

        return LLMResponse(text=text, input_tokens=len(user) // 4 or 1,
                           output_tokens=len(text) // 4 or 1,
                           model="fake-demo", cached=False)

    def complete_vision(self, system, image_png, instruction, max_tokens=4096) -> LLMResponse:
        text = '{"text": "[demo-vision] Handwritten note: meeting on Tuesday.", "confidence": 0.78}'
        return LLMResponse(text=text, input_tokens=64, output_tokens=20,
                           model="fake-demo", cached=False)

    # ---- helpers ----
    @staticmethod
    def _user_text(messages: list[dict]) -> str:
        last = messages[-1]["content"] if messages else ""
        if isinstance(last, str):
            return last
        return " ".join(part.get("text", "") for part in last if isinstance(part, dict))

    def _structured_fields(self, text: str) -> str:
        parties = re.findall(r"(Acme [A-Z][a-z]+|Beta [A-Z][a-z]+|[A-Z][a-z]+ Corp)", text)[:4]
        dates = re.findall(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b", text)[:5]
        return json.dumps({
            "doc_type": "contract" if "agreement" in text.lower() else "memo",
            "parties": parties or ["Acme Industries", "Beta Holdings"],
            "key_dates_iso": ["2026-01-15"] if not dates else [],
        })

    def _case_fact_summary(self, user_text: str) -> str:
        chunk_ids = re.findall(r"\[([A-Za-z0-9_:\-]+)\]", user_text)
        c0 = chunk_ids[0] if chunk_ids else "demo:p1:c0"
        c1 = chunk_ids[1] if len(chunk_ids) > 1 else c0
        return (
            f"## Parties\n"
            f"Acme Industries and Beta Holdings are the named parties [{c0}].\n\n"
            f"## Timeline\n"
            f"The agreement took effect on 15 January 2026 [{c1}].\n\n"
            f"## Subject Matter\n"
            f"The agreement covers the licensing of proprietary software [{c0}].\n\n"
            f"## Material Facts\n"
            f"Payment of $250,000 is due within thirty days of execution [{c1}].\n"
            f"Termination requires sixty days written notice [{c0}].\n\n"
            f"## Uncertain or Unclear\n"
            f"No supporting evidence found in the provided documents."
        )

    def _edit_signal(self, user_text: str) -> str:
        try:
            payload = json.loads(user_text)
        except Exception:
            payload = {}
        original = (payload.get("original") or "")
        edited = (payload.get("edited") or "")
        if "It should be noted" in original or "transmitted" in original:
            return json.dumps({
                "category": "boilerplate_stripped",
                "rule": "Strip 'It should be noted that' and similar boilerplate prefixes.",
                "reason": "removed throat-clearing phrase",
            })
        if "Bottom line" in edited:
            return json.dumps({
                "category": "section_added",
                "rule": "Add a one-sentence 'Bottom line' under Subject Matter.",
                "reason": "added summary line",
            })
        return json.dumps({
            "category": "phrasing_swap",
            "rule": "Prefer active voice over passive in Material Facts.",
            "reason": "rewrote passive to active",
        })

    def _simulated_edit(self, user_text: str) -> str:
        # Apply a few of the operator's house-style rules.
        text = user_text.split("Edit the following draft:", 1)[-1].strip()
        text = text.replace("It should be noted that ", "")
        text = text.replace("transmitted", "sent").replace("remitted", "paid")
        text = text.replace("instrument", "agreement")
        if "## Subject Matter" in text and "Bottom line" not in text:
            text = text.replace(
                "## Subject Matter\n",
                "## Subject Matter\nBottom line: a software licensing deal between Acme and Beta.\n",
            )
        return text


# ---------------------------------------------------------------------- Embeddings
class HashEmbedder(EmbeddingProvider):
    @property
    def name(self) -> str:
        return "fake-hash-embedder"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # 16-dim deterministic vector in [-1, 1]
            vecs.append([((b - 128) / 128.0) for b in h[:16]])
        return vecs


# ---------------------------------------------------------------------- OCR
class PassthroughOCR(OCRProvider):
    """For demo PDFs that already have a text layer, OCR is a no-op."""
    def ocr(self, image_png: bytes) -> OCRResult:
        return OCRResult(text="", confidence=0.0)


# ---------------------------------------------------------------------- Vector store
class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self._rows: dict[str, dict] = {}

    def upsert(self, items):
        for it in items:
            self._rows[it["chunk_id"]] = it
        return len(items)

    def query(self, embedding, k=5, doc_ids=None):
        rows = list(self._rows.values())
        if doc_ids:
            rows = [r for r in rows if r["doc_id"] in doc_ids]

        def sim(r):
            v = r["embedding"]
            return sum(a * b for a, b in zip(v[: len(embedding)], embedding))

        rows.sort(key=sim, reverse=True)
        return [
            StoredChunk(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"], page=r["page"],
                text=r["text"], score=0.85, ocr_confidence=r.get("ocr_confidence", 1.0),
            )
            for r in rows[:k]
        ]

    def get_text(self, chunk_id):
        r = self._rows.get(chunk_id)
        return r["text"] if r else None

    def delete_doc(self, doc_id: str) -> int:
        keys = [k for k, v in self._rows.items() if v.get("doc_id") == doc_id]
        for k in keys:
            del self._rows[k]
        return len(keys)

    def reset(self):
        self._rows.clear()


# ---------------------------------------------------------------------- wiring
def install_demo_providers() -> None:
    from providers.registry import set_embedder, set_llm, set_ocr, set_vector_store
    set_llm(FakeLLM())
    set_embedder(HashEmbedder())
    set_ocr(PassthroughOCR())
    set_vector_store(InMemoryVectorStore())
