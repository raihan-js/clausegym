"""Rule-engine tests (no DB, no model)."""

from clausegym.verify import evaluate, evaluate_text, extract_rules


def test_applies():
    r = evaluate("52.219-9", {"set_aside": "small_business", "value": 1_000_000})
    assert r["verdict"] == "applies"
    assert r["confidence"] == 1.0


def test_does_not_apply_below_threshold():
    r = evaluate("52.219-9", {"set_aside": "small_business", "value": 100_000})
    assert r["verdict"] == "does_not_apply"
    assert r["confidence"] == 0.0


def test_does_not_apply_wrong_set_aside():
    r = evaluate("52.219-8", {"set_aside": "unrestricted"})
    assert r["verdict"] == "does_not_apply"


def test_unknown_rule():
    r = evaluate("99.999-9", {})
    assert r["verdict"] == "unknown"


def test_extract_rules():
    text = "See FAR 52.219-9 and 52.204-21 for details."
    found = extract_rules(text)
    assert "52.219-9" in found
    assert "52.204-21" in found


def test_evaluate_text_end_to_end():
    text = "Applies per 52.219-8."
    out = evaluate_text(text, {"set_aside": "small_business"})
    assert out and out[0]["verdict"] == "applies"
    out2 = evaluate_text(text, {"set_aside": "unrestricted"})
    assert out2 and out2[0]["verdict"] == "does_not_apply"
