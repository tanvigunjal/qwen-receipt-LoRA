"""Throughput / latency benchmark against a running vLLM server.

Measures, per (model-mix, concurrency) cell:
  TTFT  time to first token    ~ prefill cost (reading the ~700-token receipt)
  TPOT  time per output token  ~ decode cost
  req/s and output tok/s       ~ what continuous batching buys you

Usage:
    python src/bench.py --models Qwen/Qwen3-4B-Instruct-2507             # base only
    python src/bench.py --models sroie-r16                                # one adapter
    python src/bench.py --models sroie-r8,sroie-r32                       # 2 adapters mixed
    python src/bench.py --models Qwen/Qwen3-4B-Instruct-2507,sroie-r16 --concurrency 1,8,32
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI


async def one_request(client, model, messages, max_tokens):
    t0 = time.perf_counter()
    ttft, n_tokens = None, 0
    stream = await client.chat.completions.create(
        model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
        stream=True, stream_options={"include_usage": True},
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content and ttft is None:
            ttft = time.perf_counter() - t0
        if chunk.usage:
            n_tokens = chunk.usage.completion_tokens
    total = time.perf_counter() - t0
    ttft = ttft or total
    tpot = (total - ttft) / max(n_tokens - 1, 1)
    return {"ttft": ttft, "total": total, "tokens": n_tokens, "tpot": tpot}


async def run_cell(client, models, prompts, concurrency, max_tokens):
    sem = asyncio.Semaphore(concurrency)

    async def worker(i, msgs):
        async with sem:
            # round-robin across models -> mixed-adapter batches on the server
            return await one_request(client, models[i % len(models)], msgs, max_tokens)

    t0 = time.perf_counter()
    res = await asyncio.gather(*[worker(i, p) for i, p in enumerate(prompts)])
    wall = time.perf_counter() - t0
    p50 = lambda xs: statistics.median(xs)
    return {
        "models": "+".join(models), "concurrency": concurrency, "n": len(res),
        "req_per_s": round(len(res) / wall, 2),
        "out_tok_per_s": round(sum(r["tokens"] for r in res) / wall, 1),
        "ttft_p50_ms": round(1000 * p50([r["ttft"] for r in res]), 1),
        "tpot_p50_ms": round(1000 * p50([r["tpot"] for r in res]), 2),
        "latency_p50_s": round(p50([r["total"] for r in res]), 3),
    }


async def main_async(args):
    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    prompts = [r["prompt"] for r in rows][: args.n]
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    models = args.models.split(",")

    # warm-up: loads adapters into GPU slots, compiles kernels, fills caches
    await asyncio.gather(*[one_request(client, m, prompts[0], 8) for m in models])

    results = []
    for c in [int(x) for x in args.concurrency.split(",")]:
        cell = await run_cell(client, models, prompts, c, args.max_tokens)
        results.append(cell)
        print(json.dumps(cell))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"bench_{'+'.join(m.split('/')[-1] for m in models)}.json").write_text(
        json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model / adapter names")
    ap.add_argument("--concurrency", default="1,4,16,32")
    ap.add_argument("--n", type=int, default=64, help="requests per cell")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--data", default="data/processed/test.jsonl")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--out-dir", default="results")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
