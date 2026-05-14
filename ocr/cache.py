"""Image-hash backed OCR result cache.

OCR is expensive. If the same page comes back (re-ingest, eval rerun,
demo loop), we want to return the existing result without spinning up
Tesseract again. We key on a perceptual-ish hash of the image bytes
plus the engine config so cache hits are precise.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass
from storage.db import get_conn


CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ocr_cache (
    cache_key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


@dataclass
class CachedOcr:
    text: str
    confidence: float
    engine: str
    sanity_score: float
    metadata: dict


def _ensure_table() -> None:
    get_conn().executescript(CREATE_SQL)
    get_conn().commit()


def make_key(image_bytes: bytes, engine: str, config: str) -> str:
    h = hashlib.sha256()
    h.update(image_bytes)
    h.update(engine.encode("utf-8"))
    h.update(config.encode("utf-8"))
    return h.hexdigest()


def get(key: str) -> CachedOcr | None:
    _ensure_table()
    row = get_conn().execute(
        "SELECT payload FROM ocr_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    if not row:
        return None
    p = json.loads(row["payload"])
    return CachedOcr(**p)


def put(key: str, result: CachedOcr) -> None:
    _ensure_table()
    get_conn().execute(
        "INSERT OR REPLACE INTO ocr_cache(cache_key, payload) VALUES(?, ?)",
        (key, json.dumps(asdict(result))),
    )
    get_conn().commit()
