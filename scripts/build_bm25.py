"""Build BM25 index over all chunks in Postgres. Run: python scripts/build_bm25.py."""

from __future__ import annotations

import json
from pathlib import Path

import bm25s
import psycopg

DSN = "dbname=clausegym user=postgres password=clausegym host=localhost port=5434"
BATCH = 20_000


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "bm25"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    all_tokens: list[list[int]] = []

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            total = cur.fetchone()[0]
            print(f"Indexing {total} chunks...", flush=True)
            offset = 0
            while offset < total:
                cur.execute(
                    "SELECT chunk_id, subject, text FROM chunks "
                    "ORDER BY chunk_id LIMIT %s OFFSET %s",
                    (BATCH, offset),
                )
                rows = cur.fetchall()
                if not rows:
                    break
                docs = [(r[1] + " " + r[2] if r[1] else r[2]) for r in rows]
                # return_ids=False -> plain string tokens, so batches can merge
                # into one corpus with a single vocab at index() time.
                toks = bm25s.tokenize(docs, stopwords="en", return_ids=False, show_progress=False)
                all_tokens.extend(toks)
                ids.extend(r[0] for r in rows)
                offset += len(rows)
                print(f"  tokenized {offset}/{total}", flush=True)

    print("Fitting BM25...", flush=True)
    retriever = bm25s.BM25()
    retriever.index(all_tokens)
    retriever.save(str(out_dir / "idx"))
    with open(out_dir / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"Saved index ({len(ids)} docs) to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
