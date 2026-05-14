"""FastAPI surface for the full workflow + observability endpoints + UI."""
from __future__ import annotations
import shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import settings
from errors import PSLError
from logging_setup import setup as setup_logging, get_logger
from obs import costs as costs_ledger
from obs.tracer import trace_tree
from schemas import Draft, ProcessedDocument
from ingestion.run import process
from retrieval.chunker import chunk_document
from retrieval import index
from retrieval.retriever import retrieve
from drafting.generator import generate_case_fact_summary
from edits.capture import capture_edit
from edits.rules import top_rules
from storage.db import get_conn
from storage.drafts_store import get_draft


setup_logging()
log = get_logger(__name__)

if settings.demo_mode:
    from providers.fakes import install_demo_providers
    from providers.demo_seed import seed as seed_demo
    install_demo_providers()
    seed_demo()
    log.info("demo mode enabled — fake providers installed and 2 sample docs seeded")

app = FastAPI(title="PSL Document Workflow", version="0.5.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# Force no caching on UI assets so reloads always pick up the latest version.
@app.middleware("http")
async def no_cache_ui_assets(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/ui/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


UI_DIR = Path(__file__).resolve().parent.parent / "ui"
if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


@app.get("/", include_in_schema=False)
def root():
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        return {"message": "UI not built. POST /ingest to use the API directly."}
    return FileResponse(
        index_path,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/demo-mode", include_in_schema=False)
def demo_mode():
    return {"demo_mode": settings.demo_mode}


@app.exception_handler(PSLError)
async def domain_error_handler(_, exc: PSLError):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=400, content={"error": type(exc).__name__, "detail": str(exc)}
    )


# ---- response models ---------------------------------------------------
class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    mean_confidence: float
    pages: int
    chunks_indexed: int
    doc_type: str | None


class DraftRequest(BaseModel):
    doc_ids: list[str]


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    doc_ids: list[str] | None = None


class EditRequest(BaseModel):
    edited_text: str


# ---- core pipeline -----------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)) -> IngestResponse:
    suffix = Path(file.filename or "upload").suffix
    raw_path = settings.raw_dir / (file.filename or f"upload{suffix}")
    with raw_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    doc = process(raw_path)
    (settings.processed_dir / f"{doc.doc_id}.json").write_text(
        doc.model_dump_json(indent=2), encoding="utf-8"
    )
    n = index.upsert_chunks(chunk_document(doc))
    return IngestResponse(
        doc_id=doc.doc_id,
        filename=doc.filename,
        mean_confidence=doc.mean_confidence,
        pages=len(doc.pages),
        chunks_indexed=n,
        doc_type=doc.fields.doc_type,
    )


@app.get("/documents")
def list_documents():
    docs = []
    for path in sorted(settings.processed_dir.glob("*.json")):
        try:
            d = ProcessedDocument.model_validate_json(path.read_text(encoding="utf-8"))
            docs.append({
                "doc_id": d.doc_id,
                "filename": d.filename,
                "mean_confidence": d.mean_confidence,
                "pages": len(d.pages),
                "doc_type": d.fields.doc_type,
            })
        except Exception:
            continue
    return docs


@app.get("/documents/{doc_id}", response_model=ProcessedDocument)
def get_document(doc_id: str) -> ProcessedDocument:
    path = settings.processed_dir / f"{doc_id}.json"
    if not path.exists():
        raise HTTPException(404, f"unknown doc_id: {doc_id}")
    return ProcessedDocument.model_validate_json(path.read_text(encoding="utf-8"))


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str):
    from providers.registry import get_vector_store
    text = get_vector_store().get_text(chunk_id)
    if text is None:
        raise HTTPException(404, f"unknown chunk_id: {chunk_id}")
    return {"chunk_id": chunk_id, "text": text}


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Remove a document. Drops the chunks from the vector store and deletes
    the processed JSON file. Existing drafts that reference it are left in
    place for audit.
    """
    from providers.registry import get_vector_store
    n = get_vector_store().delete_doc(doc_id)
    path = settings.processed_dir / f"{doc_id}.json"
    file_existed = path.exists()
    if file_existed:
        path.unlink()
    if n == 0 and not file_existed:
        raise HTTPException(404, f"unknown doc_id: {doc_id}")
    return {"doc_id": doc_id, "chunks_deleted": n, "file_removed": file_existed}


@app.get("/pages/{doc_id}/{page}/ocr-trace")
def get_ocr_trace(doc_id: str, page: int):
    """Return the per-page OCR diagnostics: rotation, variants, preprocessing
    stages, Tesseract vs sanity confidence, vision-fallback verdict."""
    path = settings.processed_dir / f"{doc_id}.json"
    if not path.exists():
        raise HTTPException(404, f"unknown doc_id: {doc_id}")
    doc = ProcessedDocument.model_validate_json(path.read_text(encoding="utf-8"))
    for p in doc.pages:
        if p.page == page:
            return {
                "doc_id": doc_id,
                "page": p.page,
                "source": p.source,
                "ocr_confidence": p.ocr_confidence,
                "diagnostics": p.ocr_diagnostics.model_dump() if p.ocr_diagnostics else None,
            }
    raise HTTPException(404, f"page {page} not found in {doc_id}")


@app.post("/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    hits = retrieve(req.query, k=req.k, doc_ids=req.doc_ids)
    return {"hits": [h.model_dump() for h in hits]}


@app.post("/drafts", response_model=Draft)
def create_draft(req: DraftRequest) -> Draft:
    if not req.doc_ids:
        raise HTTPException(400, "doc_ids must not be empty")
    return generate_case_fact_summary(req.doc_ids)


@app.get("/drafts/{draft_id}", response_model=Draft)
def get_draft_endpoint(draft_id: str) -> Draft:
    d = get_draft(draft_id)
    if d is None:
        raise HTTPException(404, f"unknown draft_id: {draft_id}")
    return d


@app.post("/drafts/{draft_id}/edits")
def submit_edit(draft_id: str, req: EditRequest):
    try:
        return capture_edit(draft_id, req.edited_text)
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/style-rules")
def list_style_rules(doc_type: str | None = None):
    return {"rules": top_rules(doc_type=doc_type, limit=50)}


# ---- observability endpoints -------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str):
    """Return the full span tree for a trace id (also the draft_id, hex-normalised)."""
    normalised = trace_id.replace("-", "")
    spans = trace_tree(normalised)
    if not spans:
        spans = trace_tree(trace_id)
    if not spans:
        raise HTTPException(404, f"no spans for trace {trace_id}")
    return {"trace_id": spans[0]["trace_id"], "spans": spans}


@app.get("/costs")
def get_costs(trace_id: str | None = None):
    if trace_id:
        normalised = trace_id.replace("-", "")
        return costs_ledger.totals_for_trace(normalised)
    return costs_ledger.grand_total()


@app.get("/stats")
def get_stats():
    """Quick rollup of system state for the dashboard / reviewer."""
    conn = get_conn()
    counts = {
        t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
        for t in ("drafts", "edits", "style_rules", "exemplars", "traces", "cost_ledger")
    }
    return {
        "counts": counts,
        "cost_total": costs_ledger.grand_total(),
        "settings": {
            "llm_provider": settings.active_llm_provider,
            "model": settings.active_llm_model,
            "embedding_model": settings.embedding_model,
            "ocr_threshold": settings.ocr_confidence_threshold,
        },
    }
