"""diff_pair unit tests (no DB)."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "diffs_mod", Path(__file__).resolve().parents[1] / "scripts" / "build_diffs.py"
)
diffs_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diffs_mod)
diff_pair = diffs_mod.diff_pair


def test_identical_returns_none():
    assert diff_pair("same text", "same text") is None


def test_changed_counts_positive():
    d = diff_pair("the quick brown fox", "the quick red fox jumped")
    assert d is not None
    added, removed, _, _ = d
    assert added > 0 and removed >= 0


def test_big_text_path():
    d = diff_pair("a" * 25_000, "a" * 25_000 + "b" * 100)
    assert d is not None
    added, _, _, _ = d
    assert added == 100
