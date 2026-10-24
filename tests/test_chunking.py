"""split_text regression tests (must terminate on all inputs)."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "embed_mod", Path(__file__).resolve().parents[1] / "scripts" / "embed.py"
)
embed_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(embed_mod)
split_text = embed_mod.split_text


@pytest.mark.parametrize("n", [0, 1, 2, 30, 50, 51, 100, 511, 512, 513, 600, 2000])
def test_terminates_and_covers(n):
    text = "x" * n
    chunks = split_text(text)  # would hang forever pre-fix
    if n == 0:
        assert chunks == []
        return
    assert chunks
    assert chunks[0] == text[: len(chunks[0])]
    assert text.endswith(chunks[-1])
    # every chunk within bounds, non-empty, bounded length
    for c in chunks:
        assert 0 < len(c) <= 512


def test_overlap_size():
    chunks = split_text("a" * 600)
    assert len(chunks) == 2
    assert chunks[1] == "a" * 138  # 600 - (512 - 50)


def test_short_text_single_chunk():
    assert split_text("hello world") == ["hello world"]
