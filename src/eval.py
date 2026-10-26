"""Evaluate any model served behind an OpenAI-compatible API (vLLM) on SROIE.

The same script evaluates the base model and every LoRA adapter: with vLLM an
adapter is just another value of the `model` field in the request.

Usage:
    python src/eval.py --model Qwen/Qwen3-4B-Instruct-2507 --tag base
    python src/eval.py --model Qwen/Qwen3-4B-Instruct-2507 --tag base-json --structured
    python src/eval.py --model sroie-r16 --tag lora-r16
    python src/eval.py --model sroie-r16 --tag lora-r16-json --structured
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI

from metrics import aggregate, parse_output, score_example
from schema import FIELDS, RECEIPT_SCHEMA


def _is_strict_json(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


async def predict_one(client, sem, model, ex, structured, max_tokens):
    kwargs = {}
    if structured:
        # vLLM compiles the JSON schema into a grammar and masks invalid tokens
        # at every decoding step -> output is guaranteed to match the schema.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "receipt", "schema": RECEIPT_SCHEMA},
        }
    async with sem:
        t0 = time.perf_counter()
        resp = await client.chat.completions.create(
            model=model,
            messages=ex["prompt"],
            temperature=0.0,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency = time.perf_counter() - t0
    text = resp.choices[0].message.content or ""
    pred, schema_ok = parse_output(text)
    row = {
        "id": ex["id"],
        "raw": text,
        "pred": pred,
        "gold": ex["gold"],
        "strict_json": float(_is_strict_json(text)),   # parses with no cleanup at all
        "json_valid": float(pred is not None),         # parses after stripping fences/chatter
        "schema_valid": float(schema_ok),
        "latency_s": latency,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
    }
    row.update(score_example(pred, ex["gold"]))
    return row


async def run(args):
    examples = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.max_samples:
        examples = examples[: args.max_samples]

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    sem = asyncio.Semaphore(args.concurrency)

    t0 = time.perf_counter()
    rows = await asyncio.gather(*[
        predict_one(client, sem, args.model, ex, args.structured, args.max_tokens)
        for ex in examples
    ])
    wall = time.perf_counter() - t0

    metrics = aggregate([{k: v for k, v in r.items()
                          if isinstance(v, float) and k != "latency_s"} for r in rows])
    lat = sorted(r["latency_s"] for r in rows)
    metrics.update({
        "tag": args.tag, "model": args.model, "structured": args.structured,
        "wall_s": round(wall, 2),
        "req_per_s": round(len(rows) / wall, 2),
        "latency_p50_s": round(lat[len(lat) // 2], 3),
        "latency_p95_s": round(lat[int(len(lat) * 0.95) - 1], 3),
    })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.tag}.predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / f"{args.tag}.metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n== {args.tag}  ({args.model}, structured={args.structured}) ==")
    print(f"strict_json {metrics['strict_json']:.3f}   json_valid {metrics['json_valid']:.3f}"
          f"   schema_valid {metrics['schema_valid']:.3f}")
    for f in FIELDS:
        print(f"{f:<8} exact {metrics[f + '_exact']:.3f}   sim {metrics[f + '_sim']:.3f}")
    print(f"address  ignoring punctuation {metrics['address_loose']:.3f}"
          f"   all fields (address ignoring punctuation) {metrics['all_loose']:.3f}")
    print(f"macro    exact {metrics['macro_exact']:.3f}   sim {metrics['macro_sim']:.3f}"
          f"   all-fields-correct {metrics['all_exact']:.3f}")
    print(f"{metrics['req_per_s']} req/s, p50 {metrics['latency_p50_s']}s, "
          f"p95 {metrics['latency_p95_s']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model id or LoRA adapter name")
    ap.add_argument("--tag", required=True, help="name for this run's result files")
    ap.add_argument("--data", default="data/processed/test.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--structured", action="store_true", help="use JSON-schema decoding")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
