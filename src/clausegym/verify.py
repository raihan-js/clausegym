"""Rule engine for FAR/DFARS compliance (adapted from FedProc-Ledger)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# FAR clause patterns
FAR_PATTERN = re.compile(r"(\d{2}\.\d{3}(?:-\d+)?)(?:\s+[A-Z][a-z]+)?")
DFARS_PATTERN = re.compile(r"(\d{4}\.\d{4}(?:-\d+)?)(?:\s+[A-Z][a-z]+)?")

# Rule predicates (simplified for MVP)
RULES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "52.219-9": lambda facts: (
        facts.get("set_aside") == "small_business" and facts.get("value", 0) >= 750000
    ),
    "52.219-8": lambda facts: facts.get("set_aside") == "small_business",
    "52.219-6": lambda facts: facts.get("set_aside")
    in ("small_business", "8a", "sdvosb", "hubzone"),
    "52.204-21": lambda facts: facts.get("requires_cybersecurity", False),
    "52.204-23": lambda facts: facts.get("requires_cui_protection", False),
    "52.204-25": lambda facts: facts.get("requires_ffi", False),
    "25.001": lambda facts: facts.get("domestic_only", False),
    "25.204": lambda facts: facts.get("domestic_only", False),
}

SECTION_RE = re.compile(r"(\d{2}\.\d{3}(?:-\d+)?)")


def evaluate(rule_id: str, facts: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a single FAR/DFARS clause against given facts."""
    if rule_id not in RULES:
        return {
            "verdict": "unknown",
            "rule": rule_id,
            "confidence": 0.0,
            "reason": "rule not implemented",
        }
    try:
        result = RULES[rule_id]
        applies = result(facts) if callable(result) else result
        return {
            "verdict": "applies" if applies else "does_not_apply",
            "rule": rule_id,
            "confidence": 1.0 if applies else 0.0,
            "reason": "rule matched" if applies else "conditions not met",
        }
    except Exception as e:
        return {"verdict": "error", "rule": rule_id, "confidence": 0.0, "reason": str(e)}


def extract_rules(text: str) -> list[str]:
    """Extract FAR/DFARS clause references from text."""
    return sorted(set(SECTION_RE.findall(text)))


def evaluate_text(text: str, facts: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate all FAR/DFARS references in text against facts."""
    clauses = extract_rules(text)
    return [evaluate(c, facts) for c in set(clauses)]


if __name__ == "__main__":
    # Quick test
    facts = {"set_aside": "small_business", "value": 1_000_000}
    print(evaluate("52.219-9", facts))
    print(evaluate("52.219-8", facts))