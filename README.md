# ClauseGym — a versioned-compliance benchmark for RAG (FAR/DFARS 2017–2025)

Most RAG systems treat their corpus as timeless. Regulations change: answering a 2022 question with 2024 rules is *wrong*, even when correctly cited. ClauseGym measures whether retrieval stays correct **for the asked date**.

Headline result (`results/leakage_ladder.json`, frozen 500 explicit-date + 200 change questions): a dateless hybrid RRF answers explicit-date questions with the wrong year's version **80.2% of the time at top-1**. Pinning retrieval to the asked year drives top-1 leakage to **0.000** — but breaks change questions that legitimately need two versions (both-sides 0.000).

## Corpus

- FAR/DFARS Title 48 from govinfo bulk XML, 2017–2025 (9 years × 7 vols), parsed to ~91k section-versions across 15,740 distinct sections.
- **227,449 chunks** in Postgres 17 + pgvector, each with a bge-m3 1024-dim embedding and a `valid` range (`[year-01-01, year+1-01-01)`):

| Year | Chunks |
|------|--------|
| 2017 | 24,204 |
| 2018 | 25,429 |
| 2019 | 24,048 |
| 2020 | 25,409 |
| 2021 | 25,416 |
| 2022 | 25,304 |
| 2023 | 25,719 |
| 2024 | 25,627 |
| 2025 | 26,293 |

- **9,604 changed consecutive-year section pairs** in `diffs` (`results/diff_stats_s*.json`), via rule-based difflib (no LLM).
- BM25 index over all chunks (`data/bm25/`).

## M2 — retrieval sanity baseline (n=200, `results/eval_retrieval.json`)

Queries are subject + 15 lead words from held-out chunks. Rule-based scoring (exact chunk / same section+year), no LLM judge. The query construction shares vocabulary with gold chunks, so it favors BM25 by design — a sanity check, not a leaderboard.

| Method | chunk r@5 | chunk r@20 | section r@20 |
|--------|-----------|------------|--------------|
| dense | 0.505 | 0.780 | 0.805 |
| BM25 | 0.610 | 0.995 | 1.000 |
| RRF | 0.610 | 0.955 | 0.975 |

## M3 — AsOf benchmark (`data/eval/asof_questions.jsonl`)

- 500 explicit-date questions ("As of July 1 {y0}, what does FAR section {S} say?"), sampled from changed pairs, gold = chunks of (S, y0).
- 200 change questions ("What changed in section {S} between {y0} and {y1}?"), gold = chunks of both years.
- Queries use section id + subject only — no gold-text leakage.

## M4 — leakage ladder (`results/leakage_ladder.json`)

| Arm | leak_top1 ↓ | veracc@5 ↑ | veracc@20 ↑ | change both-sides |
|-----|-------------|------------|-------------|-------------------|
| naive (dateless RRF) | 0.802 | 0.106 | 0.250 | 0.960 |
| range (year ≤ asked) | 0.522 | 0.172 | 0.364 | 0.980 |
| graph (year = asked) | 0.000 | 0.332 | 0.522 | 0.000 |
| sql (oracle section) | 0.000 | 1.000 | 1.000 | 0.000 |

Caveats, stated plainly:
- `sql` is an oracle ceiling (looks up the gold section id); it is not a retriever.
- `graph`/`sql` pin to one year, so they score 0 on change questions. Strict pinning breaks legitimate multi-version questions — a real system must detect those and fetch both sides.
- The rule verifier (`src/clausegym/verify.py`) covers 8 FAR clauses; the RL loop (GRPO) is scaffolded, not yet trained.

## Reproduce

```bash
# 1. Database
docker run -d --name clausegym-pg \
  -e POSTGRES_PASSWORD=clausegym \
  -e POSTGRES_DB=clausegym \
  -p 5434:5432 \
  pgvector/pgvector:pg17

# 2. Install
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# 3. Schema
.venv/bin/python -c "from clausegym.store import init_schema; init_schema()"

# 4. Ingest (downloads ~200MB govinfo XML) + embed (GPU recommended)
.venv/bin/python -m clausegym.ingest 2017 2025
.venv/bin/python scripts/embed.py --device cuda

# 5. BM25 + diffs + questions + evals
.venv/bin/python scripts/build_bm25.py
.venv/bin/python scripts/build_diffs.py --shard 0 --nshards 4  # x4 shards
.venv/bin/python scripts/make_asof_questions.py
.venv/bin/python scripts/eval_retrieval.py
.venv/bin/python scripts/eval_leakage.py --arm naive --offset 0 --limit 350  # x arms/offsets

# 6. Tests
.venv/bin/pytest tests/ -v
```

Environment: `ASOF_DSN` (default `dbname=clausegym user=postgres password=clausegym host=localhost port=5434`), `EMBED_MODEL` (default `BAAI/bge-m3`), `EMBED_DEVICE` (default `cuda`).

## Layout

```
clausegym/
├── src/clausegym/
│   ├── ingest.py    # govinfo XML → sections
│   ├── store.py     # Postgres + pgvector (valid ranges)
│   ├── retrieve.py  # BM25 + dense + RRF + year filters
│   └── verify.py    # rule engine (8 FAR clauses)
├── scripts/         # embed / build_bm25 / build_diffs / make_asof_questions / eval_*
├── tests/           # pytest suite
├── data/            # raw XML, sections_*.json, bm25/, eval/
└── results/         # per-run JSON (diff stats, baselines, leakage ladder)
```

## Citation

```bibtex
@software{clausegym2026,
  title = {ClauseGym: A Versioned-Compliance Benchmark for RAG (FAR/DFARS 2017-2025)},
  author = {Raihan Sikder},
  year = {2026},
  url = {https://github.com/raihan-js/clausegym},
  note = {227,449 chunks, 9,604 diffs, 700 frozen AsOf questions, rule-based scoring}
}
```

## License

Apache-2.0. Corpus: US government works (public domain). Model weights: base model licence.
