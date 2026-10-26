"""SROIE -> chat-format JSONL for TRL SFT + evaluation.

Input  (from github.com/zzzDavid/ICDAR-2019-SROIE, folder data/):
    box/XXX.csv   one OCR line per row: x1,y1,x2,y2,x3,y3,x4,y4,text
    key/XXX.json  {"company", "date", "address", "total"}

Output (data/processed/):
    train.jsonl / val.jsonl / test.jsonl, one receipt per line:
    {"id", "prompt": [system, user], "completion": [assistant], "gold": {...}, "ocr_text"}
    stats.json  dataset statistics + "how often is the gold value in the OCR text"

Usage:
    python src/data_prep.py --sroie-dir data/raw/ICDAR-2019-SROIE/data --out-dir data/processed
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

from schema import FIELDS, build_prompt, to_target


# --------------------------------------------------------------------------- #
# 1. Parse OCR boxes
# --------------------------------------------------------------------------- #
def read_boxes(path: Path) -> list[dict]:
    """Read one SROIE box file -> list of {x0, y0, x1, y1, text}."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    boxes = []
    for line in text.splitlines():
        parts = line.strip().split(",", 8)  # text itself may contain commas
        if len(parts) < 9 or not parts[8].strip():
            continue
        try:
            xs = [int(parts[i]) for i in (0, 2, 4, 6)]
            ys = [int(parts[i]) for i in (1, 3, 5, 7)]
        except ValueError:
            continue
        boxes.append({"x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys),
                      "text": parts[8].strip()})
    return boxes


# --------------------------------------------------------------------------- #
# 2. Serialize layout -> text (the "layout serialization" step)
# --------------------------------------------------------------------------- #
def serialize(boxes: list[dict], mode: str = "rows") -> str:
    """Turn 2-D boxes into 1-D text for the LLM.

    plain: one box per line, in the order the OCR engine produced them
    rows:  group boxes whose vertical centres are close into one visual row,
           left-to-right, joined with ' | '. Keeps "TOTAL ... 9.00" on one line,
           which is exactly the signal the model needs for the total field.
    """
    if mode == "plain" or not boxes:
        return "\n".join(b["text"] for b in boxes)

    heights = [b["y1"] - b["y0"] for b in boxes]
    tol = 0.5 * statistics.median(heights)

    rows: list[list[dict]] = []
    for b in sorted(boxes, key=lambda b: (b["y0"] + b["y1"]) / 2):
        cy = (b["y0"] + b["y1"]) / 2
        if rows:
            row_cy = statistics.mean((r["y0"] + r["y1"]) / 2 for r in rows[-1])
            if abs(cy - row_cy) <= tol:
                rows[-1].append(b)
                continue
        rows.append([b])

    return "\n".join(" | ".join(b["text"] for b in sorted(r, key=lambda b: b["x0"]))
                     for r in rows)


# --------------------------------------------------------------------------- #
# 3. Gold labels
# --------------------------------------------------------------------------- #
def read_key(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {f: (data.get(f) or None) for f in FIELDS}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().upper()


# --------------------------------------------------------------------------- #
# 4. Build examples, split, write
# --------------------------------------------------------------------------- #
def build_example(rid: str, ocr_text: str, gold: dict) -> dict:
    return {
        "id": rid,
        "prompt": build_prompt(ocr_text),
        "completion": [{"role": "assistant", "content": to_target(gold)}],
        "gold": gold,
        "ocr_text": ocr_text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sroie-dir", required=True, help="folder containing box/ and key/")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--layout", choices=["rows", "plain"], default="rows")
    ap.add_argument("--n-val", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.sroie_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    examples, skipped = [], 0
    for key_path in sorted((src / "key").glob("*.json")):
        box_path = src / "box" / f"{key_path.stem}.csv"
        if not box_path.exists():
            skipped += 1
            continue
        boxes = read_boxes(box_path)
        if not boxes:
            skipped += 1
            continue
        examples.append(build_example(key_path.stem, serialize(boxes, args.layout),
                                      read_key(key_path)))

    # Split by receipt id with a fixed seed -> reproducible, no leakage.
    random.Random(args.seed).shuffle(examples)
    test = examples[: args.n_test]
    val = examples[args.n_test : args.n_test + args.n_val]
    train = examples[args.n_test + args.n_val :]

    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # How often does the gold value literally appear in the OCR text?
    # This is a soft ceiling for exact match: if the label is not in the input,
    # the model has to *correct* OCR noise, not just copy.
    in_text = {f: 0 for f in FIELDS}
    present = {f: 0 for f in FIELDS}
    for ex in examples:
        flat = norm(ex["ocr_text"].replace(" | ", " ").replace("\n", " "))
        for f in FIELDS:
            v = ex["gold"][f]
            if v:
                present[f] += 1
                in_text[f] += norm(v) in flat

    chars = [len(ex["ocr_text"]) for ex in examples]
    stats = {
        "n_receipts": len(examples), "skipped": skipped, "layout": args.layout,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "ocr_chars": {"mean": round(statistics.mean(chars)), "max": max(chars)},
        "approx_prompt_tokens_max": max(chars) // 3 + 250,
        "gold_found_verbatim_in_ocr": {f: round(in_text[f] / max(present[f], 1), 3)
                                       for f in FIELDS},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
