"""M3a: consecutive-year section diffs into the diffs table.

Resumable by alphabetical section shard: --shard i/n processes sections
whose md5 hex digest falls in [i/n, (i+1)/n). Each run commits its own rows
(fresh DELETE only on shard 0). Rule-based, no LLM.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import psycopg

DSN = "dbname=clausegym user=postgres password=clausegym host=localhost port=5434"
YEARS = list(range(2017, 2026))
WS = re.compile(r"\s+")
BIG = 20_000  # texts longer than this use quick estimates, not opcodes


def norm(t: str) -> str:
    return WS.sub(" ", t).strip()


def in_shard(section: str, shard: int, nshards: int) -> bool:
    h = int(hashlib.md5(section.encode()).hexdigest(), 16)
    return (h % nshards) == shard


def diff_pair(t0_: str, t1_: str) -> tuple[int, int, str, str] | None:
    if t0_ == t1_:
        return None
    if max(len(t0_), len(t1_)) > BIG:
        sm = difflib.SequenceMatcher(None, t0_, t1_, autojunk=True)
        return abs(len(t1_) - len(t0_)), 0, t1_[:500], t0_[:500]
    sm = difflib.SequenceMatcher(None, t0_, t1_, autojunk=False)
    added = removed = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            added += j2 - j1
        if tag in ("delete", "replace"):
            removed += i2 - i1
    return added, removed, t1_[:500], t0_[:500]


def main(shard: int, nshards: int) -> None:
    t0 = time.time()
    by_section: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT section, year, chunk_id, text "
                "FROM chunks ORDER BY section, year, chunk_id"
            )
            n_rows = 0
            for section, year, _cid, text in cur:
                if in_shard(section, shard, nshards):
                    by_section[section][year].append(text)
                n_rows += 1
    print(
        f"Shard {shard}/{nshards}: {len(by_section)} sections ({n_rows} rows)",
        flush=True,
    )

    rows = []
    for s, per_year in by_section.items():
        joined = {y: norm(" ".join(parts)) for y, parts in per_year.items()}
        for y0, y1 in zip(YEARS[:-1], YEARS[1:]):
            t0_, t1_ = joined.get(y0), joined.get(y1)
            if t0_ is None or t1_ is None:
                continue
            d = diff_pair(t0_, t1_)
            if d is not None:
                added, removed, a_ex, r_ex = d
                rows.append((s, y0, y1, a_ex, r_ex, added, removed))

    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            if shard == 0:
                cur.execute("DELETE FROM diffs")
            for r in rows:
                cur.execute(
                    """INSERT INTO diffs (section, y0, y1, added, removed,
                                       chars_added, chars_removed)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (section, y0, y1) DO NOTHING""",
                    r,
                )
            conn.commit()
    print(f"Shard {shard}: {len(rows)} changed pairs in {time.time()-t0:.0f}s", flush=True)

    root = Path(__file__).resolve().parents[1]
    stat_path = root / "results" / f"diff_stats_s{shard}.json"
    stat_path.write_text(json.dumps({
        "shard": shard, "nshards": nshards,
        "sections": len(by_section), "changed_pairs": len(rows),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=4)
    a = p.parse_args()
    main(a.shard, a.nshards)
