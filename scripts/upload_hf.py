"""Upload the ClauseGym benchmark artifact to Hugging Face.

Dataset: raihan-js/clausegym-bench — frozen AsOf questions, diffs table,
leakage ladder, retrieval baseline. All text is US public domain (FAR/DFARS).
Requires HF_TOKEN in the environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import psycopg

REPO_ID = "raihan-js/clausegym-bench"
DSN = os.environ.get(
    "ASOF_DSN",
    "dbname=clausegym user=postgres password=clausegym host=localhost port=5434",
)

CARD = """\
---
license: apache-2.0
task_categories:
  - text-retrieval
tags:
  - rag
  - temporal
  - legal
  - benchmark
---

# ClauseGym-Bench

Frozen benchmark for version-aware retrieval over FAR/DFARS 2017–2025
(US public domain). Companion to https://github.com/raihan-js/clausegym.

- `asof_questions.jsonl` (700): 500 explicit-date + 200 change questions
  with gold chunk IDs. Queries use section id + subject only.
- `diffs.jsonl` (9,604): consecutive-year changed section pairs with
  character counts and 500-char excerpts.
- `leakage_ladder.json`: naive 0.802 → range 0.522 → graph 0.000 top-1
  leakage on the frozen set (rule-based scoring, no LLM judge).
- `eval_retrieval.json`: M2 sanity baseline (n=200).
- `asof_meta.json`: provenance (created_at, source rows).

Corpus behind the IDs: 227,449 bge-m3 chunks in Postgres+pgvector
(rebuild with the repo scripts; embeddings not included).
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stage = Path("/tmp/clausegym-bench")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    shutil.copy(root / "data" / "eval" / "asof_questions.jsonl", stage)
    shutil.copy(root / "data" / "eval" / "asof_meta.json", stage)
    shutil.copy(root / "results" / "leakage_ladder.json", stage)
    shutil.copy(root / "results" / "eval_retrieval.json", stage)
    (stage / "README.md").write_text(CARD)

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT section, y0, y1, chars_added, chars_removed, "
                "added, removed FROM diffs ORDER BY section, y0"
            )
            with open(stage / "diffs.jsonl", "w", encoding="utf-8") as f:
                for s, y0, y1, ca, cr, a, r in cur:
                    f.write(json.dumps({
                        "section": s, "y0": y0, "y1": y1,
                        "chars_added": ca, "chars_removed": cr,
                        "new_excerpt": a, "old_excerpt": r,
                    }) + "\n")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(stage),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="ClauseGym-Bench: 700 frozen AsOf questions, 9,604 diffs, leakage ladder",
    )
    print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}")
    subprocess.run(["rm", "-rf", str(stage)], check=False)


if __name__ == "__main__":
    main()
