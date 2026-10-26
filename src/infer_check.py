"""Sanity-check an adapter with plain Transformers + PEFT before serving it.

Answers "did training actually change the model?" in under a minute.
With --compare it also runs the base model with the adapter disabled.

Usage:
    python src/infer_check.py --adapter outputs/lora-r16 --n 3 --compare
"""

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def generate(model, tok, messages, max_new_tokens=200):
    ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base", default=None, help="default: read from adapter_config.json")
    ap.add_argument("--data", default="data/processed/val.jsonl")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--compare", action="store_true", help="also show base-model output")
    args = ap.parse_args()

    base = args.base or json.load(open(f"{args.adapter}/adapter_config.json"))[
        "base_model_name_or_path"]
    tok = AutoTokenizer.from_pretrained(base)
    dtype = ((torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
             if torch.cuda.is_available() else torch.float32)   # fp16 on T4 (no native bf16)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]
    for r in rows:
        print(f"\n=== receipt {r['id']} ===")
        print("gold :", json.dumps(r["gold"], ensure_ascii=False))
        print("lora :", generate(model, tok, r["prompt"]))
        if args.compare:
            with model.disable_adapter():          # same weights, adapter switched off
                print("base :", generate(model, tok, r["prompt"]))


if __name__ == "__main__":
    main()
