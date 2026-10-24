"""M2 rule-based retrieval baseline: dense vs BM25 vs RRF.

No LLM judge. Queries are built from held-out chunks (subject + leading
words); a hit is retrieving the source chunk (chunk-level) or any chunk of
the same section+year (section-level). Favors BM25 lexically by construction;
reported as a sanity baseline, not a leaderboard.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from clausegym.retrieve import bm25_search, dense_search, rrf

DSN = "dbname=clausegym user=postgres password=clausegym host=localhost port=5434"
N = 200
K = 50


def sample_chunks(n: int = N) -> list[dict]:
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, section, year, subject, text
                FROM chunks
                ORDER BY md5(chunk_id)
                LIMIT %s
                """,
                (n,),
            )
            return [
                {"chunk_id": r[0], "section": r[1], "year": r[2],
                 "subject": r[3] or "", "text": r[4]}
                for r in cur.fetchall()
            ]


def make_query(c: dict) -> str:
    words = c["text"].split()[:15]
    lead = " ".join(words)
    return f"{c['subject']} {lead}".strip()


def recall_at(hits: list[str], gold: str, k: int) -> int:
    return int(gold in hits[:k])


def section_recall_at(hits: list[str], section: str, year: int, k: int, meta: dict) -> int:
    for h in hits[:k]:
        s, y = meta.get(h, (None, None))
        if s == section and y == year:
            return 1
    return 0


def main() -> None:
    from clausegym.retrieve import chunk_years, get_bm25  # noqa: F401 (warm caches)

    chunks = sample_chunks()
    print(f"Sampled {len(chunks)} chunks", flush=True)
    queries = [(c, make_query(c)) for c in chunks]

    # Batch-embed all queries at once (fast on GPU)
    from clausegym.retrieve import _embedder
    model = _embedder()
    embs = model.encode([q for _, q in queries], normalize_embeddings=True, batch_size=32)

    # Chunk metadata for section-level scoring (one DB round trip)
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, section, year FROM chunks")
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    print(f"Loaded metadata for {len(meta)} chunks", flush=True)

    per_item = []
    for i, (c, q) in enumerate(queries):
        gold = c["chunk_id"]
        d = dense_search(embs[i].tolist(), k=K)
        b = bm25_search(q, k=K)
        f = rrf([b, d])[:K]
        per_item.append({
            "chunk_id": gold,
            "section": c["section"],
            "year": c["year"],
            "query": q[:200],
            "dense_r5": recall_at(d, gold, 5),
            "dense_r20": recall_at(d, gold, 20),
            "bm25_r5": recall_at(b, gold, 5),
            "bm25_r20": recall_at(b, gold, 20),
            "rrf_r5": recall_at(f, gold, 5),
            "rrf_r20": recall_at(f, gold, 20),
            "dense_s20": section_recall_at(d, c["section"], c["year"], 20, meta),
            "bm25_s20": section_recall_at(b, c["section"], c["year"], 20, meta),
            "rrf_s20": section_recall_at(f, c["section"], c["year"], 20, meta),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(queries)}", flush=True)

    def mean(key: str) -> float:
        return sum(r[key] for r in per_item) / len(per_item)

    summary = {
        "n": len(per_item),
        "k": K,
        "chunk_recall@5": {m: round(mean(f"{m}_r5"), 4) for m in ("dense", "bm25", "rrf")},
        "chunk_recall@20": {m: round(mean(f"{m}_r20"), 4) for m in ("dense", "bm25", "rrf")},
        "section_recall@20": {m: round(mean(f"{m}_s20"), 4) for m in ("dense", "bm25", "rrf")},
    }
    out = {
        "run": "m2-baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus_chunks": len(meta),
        "method": "subject+15-word queries from held-out chunks; exact chunk/section match",
        "limitation": "queries share vocabulary with gold chunks, "
                        "favoring BM25; sanity baseline only",
        "summary": summary,
        "items": per_item,
    }
    root = Path(__file__).resolve().parents[1]
    (root / "results" / "eval_retrieval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
