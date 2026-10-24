"""RRF fusion properties (no DB, no model)."""

from clausegym.retrieve import rrf


def test_empty():
    assert rrf([]) == []


def test_single_list_order_preserved():
    assert rrf([["a", "b", "c"]]) == ["a", "b", "c"]


def test_shared_doc_ranked_first():
    # 'x' at rank 0 in both lists beats anything ranked lower
    out = rrf([["x", "a"], ["b", "x"]])
    assert out[0] == "x"


def test_k_parameter():
    out60 = rrf([["a"], ["a"]])
    out01 = rrf([["a"], ["a"]], k=1)
    assert out60 == out01 == ["a"]
