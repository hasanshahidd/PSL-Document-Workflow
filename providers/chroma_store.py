"""ChromaDB vector store provider."""
from __future__ import annotations
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import settings
from obs.tracer import span
from providers.base import StoredChunk, VectorStore


COLLECTION = "legal_chunks"


class ChromaVectorStore(VectorStore):
    def __init__(self, path: str | None = None):
        self._client = chromadb.PersistentClient(
            path=str(path or settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, items: list[dict]) -> int:
        if not items:
            return 0
        with span("vectorstore.upsert", n=len(items)):
            self._collection.upsert(
                ids=[i["chunk_id"] for i in items],
                embeddings=[i["embedding"] for i in items],
                documents=[i["text"] for i in items],
                metadatas=[
                    {"doc_id": i["doc_id"], "page": i["page"],
                     "ocr_confidence": i.get("ocr_confidence", 1.0)}
                    for i in items
                ],
            )
            return len(items)

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        doc_ids: list[str] | None = None,
    ) -> list[StoredChunk]:
        with span("vectorstore.query", k=k) as s:
            where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
            res = self._collection.query(query_embeddings=[embedding], n_results=k, where=where)
            out: list[StoredChunk] = []
            for cid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            ):
                out.append(
                    StoredChunk(
                        chunk_id=cid,
                        doc_id=meta["doc_id"],
                        page=meta["page"],
                        text=doc,
                        score=1.0 - dist,
                        ocr_confidence=meta.get("ocr_confidence", 1.0),
                    )
                )
            s.set("hits", len(out))
            return out

    def get_text(self, chunk_id: str) -> str | None:
        res = self._collection.get(ids=[chunk_id])
        docs = res.get("documents") or []
        return docs[0] if docs else None

    def delete_doc(self, doc_id: str) -> int:
        with span("vectorstore.delete_doc", doc_id=doc_id) as s:
            res = self._collection.get(where={"doc_id": doc_id})
            ids = res.get("ids") or []
            if ids:
                self._collection.delete(ids=ids)
            s.set("n_deleted", len(ids))
            return len(ids)

    def reset(self) -> None:
        self._client.reset()
