"""Hybrid retrieval: BM25 + dense (RRF fusion) + version filters."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import bm25s
import psycopg

DSN = os.environ.get(
    "ASOF_DSN",
    "dbname=clausegym user=postgres password=clausegym host=localhost port=5434",
)
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = 1024
DEVICE = os.environ.get("EMBED_DEVICE", "cuda")

ROOT = Path(__file__).resolve().parents[2]
BM25_DIR = ROOT / "data" / "bm25"

_model: Any = None
_bm25: Any = None
_bm25_ids: list[str] = []


def _embedder() -> Any:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return _model


def get_bm25() -> tuple[Any, list[str]]:
    global _bm25, _bm25_ids
    if _bm25 is None:
        _bm25 = bm25s.BM25.load(str(BM25_DIR / "idx"), load_corpus=False)
        with open(BM25_DIR / "ids.json", encoding="utf-8") as f:
            _bm25_ids = json.load(f)
    return _bm25, _bm25_ids


def embed_query(text: str) -> list[float]:
    vecs = _embedder().encode([text], normalize_embeddings=True)
    return [float(v) for v in vecs[0].tolist()]


def bm25_search(query: str, k: int = 50) -> list[str]:
    retriever, ids = get_bm25()
    toks = bm25s.tokenize([query], stopwords="en", return_ids=False, show_progress=False)
    # Passing corpus= returns our chunk IDs directly instead of row numbers.
    res = retriever.retrieve(toks, corpus=ids, k=k, show_progress=False, return_as="documents")
    return list(res[0]) if not hasattr(res[0], "tolist") else res[0].tolist()


def dense_search(embedding: list[float], k: int = 50, year: int | None = None) -> list[str]:
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            if year is not None:
                cur.execute(
                    """
                    SELECT chunk_id FROM chunks
                    WHERE year = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (year, embedding, k),
                )
            else:
                cur.execute(
                    """
                    SELECT chunk_id FROM chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, k),
                )
            return [r[0] for r in cur.fetchall()]


def chunk_years(chunk_ids: list[str]) -> dict[str, int]:
    """Map chunk_ids to their year (for post-hoc year filtering)."""
    if not chunk_ids:
        return {}
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, year FROM chunks WHERE chunk_id = ANY(%s)",
                (chunk_ids,),
            )
            return {r[0]: r[1] for r in cur.fetchall()}


def rrf(rank_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for i, doc_id in enumerate(ranks):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + i + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])]


def hybrid_search(query: str, year: int | None = None, k: int = 50) -> list[str]:
    """Hybrid BM25 + dense retrieval with RRF, optional year filter."""
    bm25_ids = bm25_search(query, k=k)
    emb = embed_query(query)
    dense_ids = dense_search(emb, k=k, year=year)
    fused = rrf([bm25_ids, dense_ids])
    if year is not None:
        years = chunk_years(fused)
        fused = [c for c in fused if years.get(c) == year]
    return fused[:k]
