"""Precision@k and Mean Reciprocal Rank against a hand-labelled query set.

The labels file is a JSON list of objects:
  [{"query": str, "relevant_chunk_ids": [str, ...], "doc_ids": [str, ...] or null}, ...]
"""
from __future__ import annotations
import json
from pathlib import Path
from retrieval.retriever import retrieve


def precision_at_k(hits_ids: list[str], gold: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for h in hits_ids[:k] if h in gold) / k


def reciprocal_rank(hits_ids: list[str], gold: set[str]) -> float:
    for i, h in enumerate(hits_ids, 1):
        if h in gold:
            return 1.0 / i
    return 0.0


def evaluate(labels_path: Path, k: int = 5) -> dict:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    per_query = []
    p_sum = 0.0
    rr_sum = 0.0
    for item in labels:
        gold = set(item["relevant_chunk_ids"])
        hits = retrieve(item["query"], k=k, doc_ids=item.get("doc_ids"))
        hit_ids = [h.chunk_id for h in hits]
        p = precision_at_k(hit_ids, gold, k)
        rr = reciprocal_rank(hit_ids, gold)
        per_query.append(
            {"query": item["query"], "p_at_k": p, "reciprocal_rank": rr, "hits": hit_ids}
        )
        p_sum += p
        rr_sum += rr
    n = max(len(labels), 1)
    return {
        "n_queries": len(labels),
        "k": k,
        "precision_at_k": p_sum / n,
        "mrr": rr_sum / n,
        "per_query": per_query,
    }
