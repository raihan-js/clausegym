"""Embed chunks and upsert to pgvector."""

from __future__ import annotations

import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from clausegym.store import upsert_chunks, init_schema, connect

MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
BATCH_SIZE = 32

def split_text(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Simple chunking by character count with overlap."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks

def get_done_ids() -> set[str]:
    """Chunk IDs already embedded (embedding IS NOT NULL)."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id FROM chunks WHERE embedding IS NOT NULL")
            return {r["chunk_id"] for r in cur.fetchall()}


def main(max_chunks: int = 0, device: str = "cuda"):
    """Generate embeddings for all sections and store in pgvector.

    max_chunks: embed at most N new chunks then exit (0 = all). Resumable
    across runs via the DB (skips chunk_ids already embedded).
    """
    import time

    init_schema()

    # Resolve repo root independent of CWD
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    if not data_dir.exists():
        print("No data found. Run ingest first.", flush=True)
        return

    json_files = sorted(data_dir.glob("sections_*.json"))
    if not json_files:
        print("No section JSON files found. Run ingest first.", flush=True)
        return

    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Loaded model: {MODEL_NAME} on {device}", flush=True)

    done = get_done_ids()
    print(f"Resuming: {len(done)} chunks already embedded", flush=True)

    for json_file in json_files:
        print(f"Processing {json_file}...", flush=True)
        with open(json_file, encoding="utf-8") as f:
            sections = json.load(f)

        all_chunks = []
        for section in sections:
            text = section["text"]
            section_id = section["section"]
            year = section["year"]
            subject = section.get("subject", "")

            # Chunk the text
            text_chunks = split_text(text)

            for i, chunk in enumerate(text_chunks):
                chunk_id = f"{section_id}_{year}_{i}"
                if chunk_id in done:
                    continue
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "section": section_id,
                    "year": year,
                    "subject": subject,
                    "text": chunk,
                    "embedding": None,  # Will be filled in batches
                    "valid": f"[{year}-01-01, {year+1}-01-01)",
                    "fetched_at": "2026-01-01T00:00:00Z",
                })

        total = len(all_chunks)
        if max_chunks and total > max_chunks:
            all_chunks = all_chunks[:max_chunks]
            total = len(all_chunks)
        print(f"  {total} chunks to embed this run (skipped {len(done)} done)", flush=True)
        t0 = time.time()
        # Process in batches
        for i in range(0, total, BATCH_SIZE):
            batch = all_chunks[i:i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=BATCH_SIZE)

            for j, chunk in enumerate(batch):
                chunk["embedding"] = embeddings[j].tolist()

            upsert_chunks(batch)
            done.update(c["chunk_id"] for c in batch)
            el = time.time() - t0
            rate = (i + len(batch)) / el if el > 0 else 0
            done_n = i + len(batch)
            print(f"  [{done_n}/{total}] upserted, {el:.0f}s, {rate:.1f}/s", flush=True)

    print("Embedding complete!", flush=True)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max-chunks", type=int, default=0,
                       help="Embed at most N new chunks then exit (0 = all).")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = p.parse_args()
    main(max_chunks=args.max_chunks, device=args.device)
