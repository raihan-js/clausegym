# ClauseGym — an RL environment where federal compliance rules are the verifier

Question: when the corpus changes under you, do your answers stay correct for the date asked? 
A RAG system that treats its corpus as timeless will answer a 2022 question with 2024 rules — correctly cited, completely wrong.

Plan: `docs/PLAN.md`. Decisions: `docs/decisions.md` (append-only).

## Results

### M0 — probes (this week)
- eCFR API reachability + version depth since 2017
- Postgres + pgvector via docker, fedproc-ledger reuse
- FedProc-Ledger rule engine reuse check

### M1 — versioned corpus (week 1)
- FAR/DFARS 2017–2025 from govinfo bulk XML (9 years × 7 vols/yr)
- Bitemporal Postgres: `valid` range + `fetched_at`; exclusion constraint on (section, valid)
- Diff table per section between consecutive years
- eCFR gives exact change dates; govinfo gives text
- 15k sections × 9 years = ~135k versions; diff table = ~100k rows

### M2 — production RAG baseline (week 2)
- Embeddings: bge-m3 (fast, multilingual, good for legal)
- Hybrid retrieval: BM25 + dense (RRF), then re-rank with cross-encoder
- Chunking: structure-aware (section → paragraph → sentence)
- Metrics: LLM-as-judge on 200 held-out Qs (accuracy, citation accuracy, hallucination rate)

### M3 — AsOf benchmark (week 3)
- Questions generated from section diffs (explicit date, implicit date, change questions)
- "As of DATE, what does section X say?" — version leakage = wrong version cited
- Metrics: version accuracy, version leakage rate, citation accuracy

### M4 — six arms (RL arms, one week)
| Arm | Description | Expected |
|-------|-------------|----------|
| naive | RRF, no date | high leakage |
| range-filtered | year ≤ asked | less leakage |
| version-graph | valid_on(query_date) filter | lower leakage |
| sql-first | rule-IDs → version table | lowest leakage |

### M5 — reward hacking & report (week 6)
- red-team the verifier
- leakage audit with strict verifier
- report with leakage ladder

---

## Tech stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Corpus | FAR/DFARS (govinfo bulk XML) + eCFR API | Official, versioned, free |
| DB | Postgres 17 + pgvector (bitemporal ranges) | Exclusion constraint on (section, valid) |
| Embeddings | bge-m3 (fast, multilingual, 1024-dim) | Good on legal, supports EN/JA |
| Retrieval | BM25 + dense (RRF), cross-encoder rerank | Hybrid proven on legal |
| Generator | Llama-3.2-3B-Instruct (4-bit) | Fits 12GB, good at legal |
| RL | TRL GRPO (GRPO faster, no critic) | Standard for RLVR |
| Verifier | Rule engine from FedProc-Ledger | Deterministic, auditable |

---

## Compute budget

- RTX 3060 (12 GB) + Docker
- Embeddings: ~200k chunks × 1024 → 200M vecs, fits in pgvector
- Generator: Llama-3.2-3B-Instruct (4-bit) — fits 12GB with KV cache
- RL: GRPO on 3B or 7B with LoRA (QLoRA)

---

## Schedule

| Week | Milestone |
|------|-----------|
| 1 | M0 probes + M1 corpus |
| 2 | M2 baseline RAG |
| 3 | M3 AsOf benchmark |
| 4 | M4 arms + red team |
| 5 | Report + release |

---

## Repo layout

```
clausegym/
├── src/clausegym/
│   ├── __init__.py
│   ├── ingest/       # govinfo XML → chunks + embeddings
│   ├── store/        # Postgres + pgvector
│   ├── retrieve/     # BM25 + dense + RRF
│   ├── verify/       # rule engine (from FedProc-Ledger)
│   ├── generate/     # generator + RL loop
│   └── eval/         # metrics, leakage audit
├── scripts/
│   ├── ingest.py
│   ├── embed.py
│   ├── train.py
│   └── eval.py
├── tests/
├── docs/
│   ├── PLAN.md
│   └── decisions.md
├── data/
├── results/
├── scripts/
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Open questions

1. Which LLM for generator? (Llama-3.2-3B vs Qwen2.5-3B vs local Phi-3.5)
2. SFT before RL? (yes, format first)
3. Reward: rule-based only or +format?
4. GRPO vs PPO? (GRPO preferred, no critic)

---

*Last updated: 2026-10-22*