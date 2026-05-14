"""Seed two synthetic documents into the demo so the UI is usable on first load.

Writes the ProcessedDocument JSON to disk and indexes the chunks into the
in-memory demo vector store, mirroring what the real ingest pipeline would
produce.
"""
from __future__ import annotations
from config import settings
from schemas import PageContent, ProcessedDocument, StructuredFields
from retrieval.chunker import chunk_document
from retrieval import index


DOC_A_TEXT = """LICENSING AGREEMENT

This Licensing Agreement is entered into on 15 January 2026 by and between
Acme Industries (the "Licensor") and Beta Holdings (the "Licensee").

1. Subject Matter. The Licensor grants the Licensee a non-exclusive license
   to use the proprietary software known as Helios v3 for internal business
   purposes.

2. Term. The term of this agreement begins on the effective date and runs
   for thirty-six months unless terminated earlier under Section 5.

3. Consideration. The Licensee shall pay the Licensor the sum of two
   hundred and fifty thousand US dollars ($250,000), payable within thirty
   days of execution.

4. Confidentiality. Each party shall keep confidential any non-public
   information disclosed by the other party in connection with this agreement.

5. Termination. Either party may terminate this agreement upon sixty days
   prior written notice for any reason. It should be noted that termination
   does not relieve the Licensee of accrued payment obligations.
"""


DOC_B_TEXT = """NOTICE OF DISPUTE

Matter No. PSL-2026-0418
Date: 12 March 2026

To: Beta Holdings, Legal Department
From: Counsel for Acme Industries

This notice is transmitted pursuant to Section 9 of the Licensing Agreement
dated 15 January 2026 between Acme Industries and Beta Holdings.

It should be noted that the Licensee has failed to remit the payment of
$250,000 due under Section 3 of the agreement. As stated above, payment was
due within thirty days of execution and remains outstanding ninety-one days
after that deadline.

Acme Industries hereby demands payment in full within fourteen days of
receipt of this notice. In light of the foregoing, failure to cure this
default will result in formal termination of the agreement and pursuit of
all available remedies.
"""


def _build_doc(doc_id: str, filename: str, text: str, doc_type: str,
               parties: list[str], dates: list[str], matter_id: str | None = None) -> ProcessedDocument:
    return ProcessedDocument(
        doc_id=doc_id,
        filename=filename,
        pages=[PageContent(page=1, text=text, ocr_confidence=0.92, source="digital")],
        fields=StructuredFields(
            doc_type=doc_type, parties=parties, dates=dates, matter_id=matter_id,
        ),
        full_text=text,
    )


def seed() -> list[ProcessedDocument]:
    docs = [
        _build_doc(
            "demo_licensing_agreement",
            "licensing_agreement.pdf",
            DOC_A_TEXT,
            "contract",
            ["Acme Industries", "Beta Holdings"],
            ["15 January 2026"],
        ),
        _build_doc(
            "demo_notice_of_dispute",
            "notice_of_dispute.pdf",
            DOC_B_TEXT,
            "notice",
            ["Acme Industries", "Beta Holdings"],
            ["15 January 2026", "12 March 2026"],
            matter_id="PSL-2026-0418",
        ),
    ]
    for d in docs:
        (settings.processed_dir / f"{d.doc_id}.json").write_text(
            d.model_dump_json(indent=2), encoding="utf-8"
        )
        index.upsert_chunks(chunk_document(d))
    return docs
