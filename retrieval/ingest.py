"""CLI: index a processed document into ChromaDB.

  python -m retrieval.ingest data/processed/<doc>.json
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from schemas import ProcessedDocument
from retrieval.chunker import chunk_document
from retrieval import index


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="path to a processed document JSON")
    args = p.parse_args()

    doc = ProcessedDocument.model_validate_json(args.path.read_text(encoding="utf-8"))
    chunks = chunk_document(doc)
    n = index.upsert_chunks(chunks)
    print(f"indexed {n} chunks from {doc.doc_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
