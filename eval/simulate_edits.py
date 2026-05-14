"""Synthetic operator that produces realistic edits for evaluation."""
from __future__ import annotations
from providers.registry import get_llm


OPERATOR_PERSONA = """You are a senior associate at Pearson Specter Litt reviewing a
Case Fact Summary. Your house style:

1. Lead the 'Material Facts' section with the central dispute, not the parties.
2. Strip boilerplate phrases like 'It should be noted that', 'As stated above',
   and 'In light of the foregoing'.
3. Replace passive constructions with active voice wherever possible.
4. Keep citations exactly as provided — do not invent or remove [chunk_id] tags.
5. Add a one-sentence 'Bottom line' under 'Subject Matter' that names the
   transaction or dispute in plain English.
6. Prefer 'agreement' over 'instrument', 'paid' over 'remitted', 'sent' over
   'transmitted'.

Return ONLY the edited draft. Preserve the section headings exactly.
"""


def simulate_edit(draft_text: str) -> str:
    resp = get_llm().complete(
        system=OPERATOR_PERSONA,
        messages=[{"role": "user", "content": f"Edit the following draft:\n\n{draft_text}"}],
        max_tokens=2048,
    )
    return resp.text.strip()
