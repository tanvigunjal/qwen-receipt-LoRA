"""Evaluate the base model and a LoRA adapter with plain Transformers (no vLLM needed).

Added for the Colab notebook. Uses the same prompts (from the JSONL), the same metrics
(metrics.py) and writes the same result files as eval.py, so report.py compares them.
The base model is evaluated by switching the adapter off on the same loaded weights.

Usage:
    python src/hf_eval.py --adapter outputs/lora-r16 --n 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics import aggregate, parse_output, score_example
from schema import FIELDS


def is_strict_json(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def generate_batch(model, tok, batch, max_new_tokens):
    texts = [tok.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=False)
             for ex in batch]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    n_in = enc["input_ids"].shape[1]   # left padding -> every answer starts at n_in
    return [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]


def evaluate(model, tok, examples, tag, model_name, args):
    rows = []
    t0 = time.perf_counter()
    for i in range(0, len(examples), args.batch_size):
        batch = examples[i : i + args.batch_size]
        tb = time.perf_counter()
        outs = generate_batch(model, tok, batch, args.max_new_tokens)
        per_req = (time.perf_counter() - tb) / len(batch)
        for ex, text in zip(batch, outs):
            pred, schema_ok = parse_output(text)
            row = {"id": ex["id"], "raw": text, "pred": pred, "gold": ex["gold"],
                   "strict_json": float(is_strict_json(text)),
                   "json_valid": float(pred is not None),
                   "schema_valid": float(schema_ok),
                   "latency_s": per_req}
            row.update(score_example(pred, ex["gold"]))
            rows.append(row)
        print(f"  {tag}: {len(rows)}/{len(examples)}", flush=True)
    wall = time.perf_counter() - t0

    metrics = aggregate([{k: v for k, v in r.items()
                          if isinstance(v, float) and k != "latency_s"} for r in rows])
    metrics.update({"tag": tag, "model": model_name, "structured": False, "engine": "transformers",
                    "wall_s": round(wall, 2), "req_per_s": round(len(rows) / wall, 2)})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{tag}.predictions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / f"{tag}.metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\n== {tag} ==")
    print(f"strict_json {metrics['strict_json']:.3f}   json_valid {metrics['json_valid']:.3f}"
          f"   schema_valid {metrics['schema_valid']:.3f}")
    for f in FIELDS:
        print(f"{f:<8} exact {metrics[f + '_exact']:.3f}   sim {metrics[f + '_sim']:.3f}")
    print(f"address  ignoring punctuation {metrics['address_loose']:.3f}"
          f"   all fields (address ignoring punctuation) {metrics['all_loose']:.3f}")
    print(f"macro    exact {metrics['macro_exact']:.3f}   sim {metrics['macro_sim']:.3f}"
          f"   all-fields-correct {metrics['all_exact']:.3f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=None, help="default: read from adapter_config.json")
    ap.add_argument("--data", default="data/processed/test.jsonl")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--tag", default=None, help="default: <adapter folder>-hf")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    base = args.base or json.load(open(f"{args.adapter}/adapter_config.json"))[
        "base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"             # decoder-only batching: pad on the left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = ((torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
             if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    examples = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]
    tag = args.tag or f"{Path(args.adapter).name}-hf"

    if not args.skip_base:
        with model.disable_adapter():
            evaluate(model, tok, examples, "base-hf", base, args)
    evaluate(model, tok, examples, tag, args.adapter, args)


if __name__ == "__main__":
    main()
