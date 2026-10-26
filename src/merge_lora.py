"""Merge a LoRA adapter into the base weights: W_merged = W + (alpha/r) * B @ A.

Merged  -> one normal model, zero adapter overhead, but one full copy per task.
Adapter -> one base model shared by many tasks, small per-request overhead.
Serve both with vLLM and compare with bench.py to see the trade-off.

Usage:
    python src/merge_lora.py --adapter outputs/lora-r16 --out outputs/merged-r16
"""

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()

    base = args.base or json.load(open(f"{args.adapter}/adapter_config.json"))[
        "base_model_name_or_path"]
    # Merge in 16-bit on the original (non-quantised) weights, even after QLoRA.
    dtype = ((torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
             if torch.cuda.is_available() else torch.float32)   # fp16 on T4 (no native bf16)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype)
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(base).save_pretrained(args.out)
    print(f"merged model saved to {args.out}")


if __name__ == "__main__":
    main()
