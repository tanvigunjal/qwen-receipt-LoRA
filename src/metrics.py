"""Metrics for structured extraction.

Three levels, from "is it usable at all" to "is every field right":
  1. json_valid      the output parses as JSON
  2. schema_valid    it has exactly the 4 keys, all string-or-null
  3. per-field       exact match (after light normalisation) and fuzzy similarity
  4. address_loose   address compared on letters and digits only, because SROIE address
                     labels are inconsistent about punctuation (trailing ".", "," spacing);
                     all_loose = all four fields right with the address scored this way
"""

import json
import re

from rapidfuzz.distance import Levenshtein

from schema import FIELDS


def parse_output(text: str) -> tuple[dict | None, bool]:
    """Return (parsed_dict_or_None, schema_ok). Tolerates ```json fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first {...} block (base models often add chatter).
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None, False
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None, False
    if not isinstance(obj, dict):
        return None, False
    schema_ok = set(obj) == set(FIELDS) and all(
        v is None or isinstance(v, str) for v in obj.values())
    return obj, schema_ok


def normalize(field: str, value) -> str:
    if value is None:
        return ""
    s = re.sub(r"\s+", " ", str(value)).strip().upper()
    if field == "total":
        # "RM 9.00", "9.00", "9" -> "9.00"
        num = re.sub(r"[^\d.]", "", s.replace(",", ""))
        try:
            return f"{float(num):.2f}"
        except ValueError:
            return s
    return s


def similarity(a: str, b: str) -> float:
    """1 - normalised Levenshtein distance (ANLS-style, no threshold)."""
    if not a and not b:
        return 1.0
    return 1.0 - Levenshtein.normalized_distance(a, b)


def score_example(pred: dict | None, gold: dict) -> dict:
    out = {}
    for f in FIELDS:
        p = normalize(f, (pred or {}).get(f))
        g = normalize(f, gold.get(f))
        out[f"{f}_exact"] = float(p == g)
        out[f"{f}_sim"] = similarity(p, g)
    out["all_exact"] = float(all(out[f"{f}_exact"] for f in FIELDS))

    alnum = lambda s: re.sub(r"[^A-Z0-9]", "", s)
    pa = normalize("address", (pred or {}).get("address"))
    ga = normalize("address", gold.get("address"))
    out["address_loose"] = float(alnum(pa) == alnum(ga))
    out["all_loose"] = float(out["address_loose"] == 1.0 and
                             all(out[f"{f}_exact"] for f in FIELDS if f != "address"))
    return out


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    agg = {k: round(sum(r[k] for r in rows) / n, 4) for k in keys}
    agg["macro_exact"] = round(sum(agg[f"{f}_exact"] for f in FIELDS) / len(FIELDS), 4)
    agg["macro_sim"] = round(sum(agg[f"{f}_sim"] for f in FIELDS) / len(FIELDS), 4)
    agg["n"] = n
    return agg
