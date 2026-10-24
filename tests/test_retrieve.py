"""Retrieval smoke tests (needs Postgres :5434, BM25 index, GPU model)."""

import psycopg
import pytest

from clausegym.retrieve import BM25_DIR, DSN, chunk_years, hybrid_search

try:
    psycopg.connect(DSN).close()
    ok_db = (BM25_DIR / "ids.json").exists()
except Exception:
    ok_db = False

pytestmark = pytest.mark.skipif(not ok_db, reason="DB or BM25 index unavailable")


def test_hybrid_returns_k_ids():
    hits = hybrid_search("small business set-aside", k=5)
    assert len(hits) == 5
    assert len(set(hits)) == 5


def test_hybrid_year_filter_respected():
    hits = hybrid_search("small business set-aside", year=2020, k=10)
    assert hits
    years = chunk_years(hits)
    assert all(y == 2020 for y in years.values())
