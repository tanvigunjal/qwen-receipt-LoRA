"""Collect results/*.metrics.json into one comparison table + show errors.

Usage:
    python src/report.py                     # comparison table (markdown)
    python src/report.py --errors lora-r16   # worst predictions of one run
"""

import argparse
import json
from pathlib import Path

from schema import FIELDS

COLS = ["strict_json", "json_valid", "schema_valid"] + [f"{f}_exact" for f in FIELDS] + \
       ["address_loose", "macro_exact", "macro_sim", "all_exact", "all_loose",
        "req_per_s", "latency_p50_s"]


def table(results_dir: Path):
    runs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.metrics.json"))]
    if not runs:
        print("no results yet")
        return
    print("| run | " + " | ".join(COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for r in runs:
        print(f"| {r['tag']} | " + " | ".join(str(r.get(c, "")) for c in COLS) + " |")


def errors(results_dir: Path, tag: str, k: int):
    rows = [json.loads(l) for l in open(results_dir / f"{tag}.predictions.jsonl")]
    rows.sort(key=lambda r: sum(r[f"{f}_sim"] for f in FIELDS))
    for r in rows[:k]:
        print(f"\n--- {r['id']}  json_valid={r['json_valid']}")
        for f in FIELDS:
            if not r[f"{f}_exact"]:
                p = (r["pred"] or {}).get(f)
                print(f"  {f:<8} pred={p!r}\n  {'':<8} gold={r['gold'][f]!r}")
        if r["pred"] is None:
            print("  raw:", r["raw"][:300])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--errors", help="run tag to inspect")
    ap.add_argument("-k", type=int, default=10)
    a = ap.parse_args()
    if a.errors:
        errors(Path(a.results_dir), a.errors, a.k)
    else:
        table(Path(a.results_dir))
