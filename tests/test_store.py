"""Store round-trip tests (needs local Postgres on :5434)."""

import uuid

import psycopg
import pytest

from clausegym.store import DSN, init_schema, upsert_chunks

try:
    psycopg.connect(DSN).close()
    NEEDS_DB = False
except Exception:
    NEEDS_DB = True

pytestmark = pytest.mark.skipif(NEEDS_DB, reason="Postgres :5434 unreachable")


def _chunk(**kw):
    base = {
        "chunk_id": f"t-{uuid.uuid4().hex[:8]}",
        "section": "99.999",
        "year": 2020,
        "subject": "test",
        "text": "hello world",
        "embedding": [0.01] * 1024,
        "valid": "[2020-01-01, 2021-01-01)",
        "fetched_at": "2026-01-01T00:00:00Z",
    }
    base.update(kw)
    return base


def test_upsert_and_refetch():
    init_schema()
    c = _chunk()
    assert upsert_chunks([c]) == 1
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT text FROM chunks WHERE chunk_id = %s", (c["chunk_id"],))
            assert cur.fetchone()[0] == "hello world"
            cur.execute("DELETE FROM chunks WHERE chunk_id = %s", (c["chunk_id"],))
            conn.commit()


def test_upsert_idempotent_on_chunk_id():
    c = _chunk(text="v1")
    upsert_chunks([c])
    c["text"] = "v2"
    upsert_chunks([c])
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT text FROM chunks WHERE chunk_id = %s", (c["chunk_id"],))
            assert cur.fetchone()[0] == "v2"
            cur.execute("DELETE FROM chunks WHERE chunk_id = %s", (c["chunk_id"],))
            conn.commit()


def test_same_section_year_multiple_chunks_allowed():
    a, b = _chunk(), _chunk(section="99.999")
    a["chunk_id"] += "-a"
    b["chunk_id"] += "-b"
    upsert_chunks([a, b])  # must not raise exclusion violation
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM chunks WHERE chunk_id IN (%s, %s)",
                (a["chunk_id"], b["chunk_id"]),
            )
            conn.commit()
