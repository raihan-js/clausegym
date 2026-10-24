"""M3b: frozen AsOf question set from the diffs table. Rule-based, no LLM.

Two types:
- explicit: "As of {y0}-07-01, what does FAR section {S} say?" gold=(S,y0)
- change: "What changed in FAR section {S} between {y0} and {y1}?" gold=(S,y0)+(S,y1)

Queries use section id + subject only (no gold-text leakage).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import psycopg

DSN = "dbname=clausegym user=postgres password=clausegym host=localhost port=5434"
N_EXPLICIT = 500
N_CHANGE = 200


def main() -> None:
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.section, d.y0, d.y1, MIN(c.subject)
                FROM diffs d JOIN chunks c
                  ON c.section = d.section AND c.year = d.y0
                GROUP BY d.section, d.y0, d.y1
                ORDER BY md5(d.section)
                """
            )
            pairs = cur.fetchall()
            subjects: dict[tuple[str, int], str] = {}
            cur.execute("SELECT DISTINCT section, year, subject FROM chunks")
            for s, y, subj in cur.fetchall():
                subjects.setdefault((s, y), subj or "")
            gold: dict[tuple[str, int], list[str]] = {}
            cur.execute("SELECT section, year, chunk_id FROM chunks ORDER BY chunk_id")
            for s, y, cid in cur.fetchall():
                gold.setdefault((s, y), []).append(cid)

    questions = []
    for s, y0, y1, subj in pairs[: N_EXPLICIT + N_CHANGE]:
        subj = subj or subjects.get((s, y0), "")
        if len(questions) < N_EXPLICIT:
            questions.append({
                "qid": f"exp-{len(questions):04d}",
                "type": "explicit",
                "section": s, "asked_year": y0,
                "asked_date": f"{y0}-07-01",
                "query": f"As of July 1 {y0}, what does FAR section {s} {subj} say?".strip(),
                "gold_chunks": gold.get((s, y0), []),
            })
        else:
            questions.append({
                "qid": f"chg-{len(questions) - N_EXPLICIT:04d}",
                "type": "change",
                "section": s, "y0": y0, "y1": y1,
                "query": f"What changed in FAR section {s} {subj} between {y0} and {y1}?".strip(),
                "gold_chunks": gold.get((s, y0), []) + gold.get((s, y1), []),
                "gold_y0": gold.get((s, y0), []),
                "gold_y1": gold.get((s, y1), []),
            })

    out_dir = Path(__file__).resolve().parents[1] / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_explicit": sum(1 for q in questions if q["type"] == "explicit"),
        "n_change": sum(1 for q in questions if q["type"] == "change"),
        "diff_source_rows": len(pairs),
    }
    (out_dir / "asof_meta.json").write_text(json.dumps(meta, indent=2))
    with open(out_dir / "asof_questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
