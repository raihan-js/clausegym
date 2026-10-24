"""PostgreSQL + pgvector store with bitemporal ranges."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from collections.abc import Iterator

import psycopg
from psycopg.rows import dict_row


DSN = os.environ.get(
    "ASOF_DSN",
    "dbname=clausegym user=postgres password=clausegym host=localhost port=5434",
)


@contextmanager
def connect() -> Iterator[Any]:
    conn = psycopg.connect(DSN, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def init_schema() -> None:
    """Create tables with bitemporal ranges and exclusion constraint."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS btree_gist;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    section TEXT NOT NULL,
                    year INT NOT NULL,
                    subject TEXT,
                    text TEXT NOT NULL,
                    embedding VECTOR(1024),
                    valid TSTZRANGE NOT NULL,
                    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks USING hnsw (embedding vector_cosine_ops);
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS section_versions (
                    section TEXT PRIMARY KEY,
                    valid TSTZRANGE NOT NULL,
                    text TEXT NOT NULL,
                    subject TEXT,
                    fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    EXCLUDE USING gist (section WITH =, valid WITH &&)
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS diffs (
                    section TEXT NOT NULL,
                    y0 INT NOT NULL,
                    y1 INT NOT NULL,
                    added TEXT,
                    removed TEXT,
                    chars_added INT,
                    chars_removed INT,
                    PRIMARY KEY (section, y0, y1)
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks(year);")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
                "ON chunks USING hnsw (embedding vector_cosine_ops);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_section_versions_valid "
                "ON section_versions USING gist (valid);"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_diffs_section ON diffs(section);")
            conn.commit()


def upsert_chunks(chunks: list[dict[str, Any]]) -> int:
    """Insert or update chunks (idempotent on chunk_id)."""
    with connect() as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                cur.execute("""
                    INSERT INTO chunks (chunk_id, section, year, subject,
                                        text, embedding, valid, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        fetched_at = EXCLUDED.fetched_at
                """, (
                    chunk["chunk_id"], chunk["section"], chunk["year"],
                    chunk.get("subject", ""), chunk["text"],
                    chunk["embedding"], chunk["valid"], chunk["fetched_at"]
                ))
        conn.commit()
    return len(chunks)


def search_similar(
    embedding: list[float], k: int = 10, year: int | None = None
) -> list[dict[str, Any]]:
    """Cosine similarity search with optional year filter."""
    with connect() as conn:
        with conn.cursor() as cur:
            if year is not None:
                cur.execute("""
                    SELECT chunk_id, section, year, subject, text, 1 - (embedding <=> %s) AS score
                    FROM chunks
                    WHERE year = %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """, (embedding, year, embedding, k))
            else:
                cur.execute("""
                    SELECT chunk_id, section, year, subject, text, 1 - (embedding <=> %s) AS score
                    FROM chunks
                    ORDER BY embedding <=> %s
                    LIMIT %s
                """, (embedding, embedding, k))
            rows: list[dict[str, Any]] = list(cur.fetchall())
            return rows


def get_section_versions(section: str) -> list[dict[str, Any]]:
    """Return all versions of a section with their valid ranges."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT section, lower(valid) as valid_from, upper(valid) as valid_to, subject, text
                FROM section_versions
                WHERE section = %s
                ORDER BY lower(valid)
            """, (section,))
            rows: list[dict[str, Any]] = list(cur.fetchall())
            return rows


def compute_diffs() -> list[dict[str, Any]]:
    """Diffs live in scripts/build_diffs.py; kept here as a typed stub."""
    return []


if __name__ == "__main__":
    init_schema()
    print("Schema initialized")
