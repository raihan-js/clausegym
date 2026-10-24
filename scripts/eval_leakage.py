"""M3b/M4: version-leakage eval for retrieval arms. Rule-based, no LLM.

Arms:
- naive: hybrid RRF, no date handling
- range: hybrid, dense restricted to year <= asked; BM25 post-filtered by year
- graph: dense restricted to asked year; BM25 post-filtered to asked year
- sql: gold section lookup restricted to asked year (oracle section, tests filter)

Usage: python scripts/eval_leakage.py --arm naive --limit 350 --offset 0
Writes results/leakage_<arm>_<offset>.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import psycopg

from clausegym.retrieve import bm25_search, chunk_years, dense_search, rrf

DSN = "dbname=clausegym user=postgres password=clausegym host=localhost port=5434"
K = 20


def load_questions() -> list[dict]:
    root = Path(__file__).resolve().parents[1]
    with open(root / "data" / "eval" / "asof_questions.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_arm(arm: str, query: str, emb: list[float], asked_year: int, section: str) -> list[str]:
    if arm == "naive":
        b = bm25_search(query, k=K)
        d = dense_search(emb, k=K)
        return rrf([b, d])[:K]
    if arm == "range":
        b = bm25_search(query, k=100)
        years = chunk_years(b)
        b = [c for c in b if years.get(c, 9999) <= asked_year][:K]
        d = dense_search(emb, k=K)  # pgvector has no <= filter here; filter post-hoc
        years_d = chunk_years(d)
        d = [c for c in d if years_d.get(c, 9999) <= asked_year]
        return rrf([b, d])[:K]
    if arm == "graph":
        b = bm25_search(query, k=200)
        years = chunk_years(b)
        b = [c for c in b if years.get(c) == asked_year][:K]
        d = dense_search(emb, k=K, year=asked_year)
        return rrf([b, d])[:K]
    if arm == "sql":
        with psycopg.connect(DSN) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id FROM chunks "
                    "WHERE section = %s AND year = %s ORDER BY chunk_id LIMIT %s",
                    (section, asked_year, K),
                )
                return [r[0] for r in cur.fetchall()]
    raise ValueError(arm)


def main(arm: str, offset: int, limit: int) -> None:
    from clausegym.retrieve import _embedder
    questions = load_questions()[offset: offset + limit if limit else None]
    print(f"Arm={arm} questions {offset}..{offset + len(questions)}", flush=True)

    model = _embedder()
    embs = model.encode([q["query"] for q in questions], normalize_embeddings=True, batch_size=32)

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT chunk_id, section, year FROM chunks")
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    items = []
    for i, q in enumerate(questions):
        asked = q["asked_year"] if q["type"] == "explicit" else q["y1"]
        hits = run_arm(arm, q["query"], embs[i].tolist(), asked, q["section"])
        top1 = meta.get(hits[0], (None, None)) if hits else (None, None)
        if q["type"] == "explicit":
            gold_set = set(q["gold_chunks"])
            v5 = 0
            for h in hits[:5]:
                s, y = meta.get(h, (None, None))
                if y == asked and s == q["section"]:
                    v5 = 1
                    break
            items.append({
                "qid": q["qid"], "type": "explicit",
                "leak_top1": int(top1[1] != asked) if hits else 1,
                "top1_year": top1[1],
                "asked_year": asked,
                "veracc@5": v5,
                "veracc@20": int(any(h in gold_set for h in hits)),
            })
        else:
            years_hit = {meta.get(h, (None, None))[1] for h in hits}
            items.append({
                "qid": q["qid"], "type": "change",
                "has_y0": int(q["y0"] in years_hit),
                "has_y1": int(q["y1"] in years_hit),
                "both_sides": int(q["y0"] in years_hit and q["y1"] in years_hit),
            })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(questions)}", flush=True)

    exp = [r for r in items if r["type"] == "explicit"]
    chg = [r for r in items if r["type"] == "change"]
    summary = {}
    if exp:
        summary["explicit"] = {
            "n": len(exp),
            "leak_top1": round(sum(r["leak_top1"] for r in exp) / len(exp), 4),
            "veracc@5": round(sum(r["veracc@5"] for r in exp) / len(exp), 4),
            "veracc@20": round(sum(r["veracc@20"] for r in exp) / len(exp), 4),
        }
    if chg:
        summary["change"] = {
            "n": len(chg),
            "has_y0": round(sum(r["has_y0"] for r in chg) / len(chg), 4),
            "has_y1": round(sum(r["has_y1"] for r in chg) / len(chg), 4),
            "both_sides": round(sum(r["both_sides"] for r in chg) / len(chg), 4),
        }
    out = {
        "arm": arm, "offset": offset, "k": K,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "items": items,
    }
    root = Path(__file__).resolve().parents[1]
    (root / "results" / f"leakage_{arm}_{offset}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="naive")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    main(a.arm, a.offset, a.limit)
