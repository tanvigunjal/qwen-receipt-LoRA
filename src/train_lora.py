"""LoRA / QLoRA supervised fine-tuning of Qwen on SROIE with TRL.

Usage:
    # bf16 LoRA (24 GB GPU: RTX 4090 / L4 / A10)
    python src/train_lora.py --rank 16 --out outputs/lora-r16

    # QLoRA (16 GB GPU, e.g. Colab T4): base model in 4-bit, adapter in 16-bit
    python src/train_lora.py --rank 16 --qlora --out outputs/lora-r16

Experiments worth running (one flag each):
    --rank 8 / 16 / 32                capacity of the adapter
    --targets attn | all              which linear layers get adapters
    --full-loss                       also train on the prompt tokens
    --epochs 1 / 3                    under- vs over-fitting on 476 receipts
"""

import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

TARGETS = {
    # attention projections only (the original LoRA paper setting)
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # every linear layer in the transformer blocks (attention + MLP)
    "all": "all-linear",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--train", default="data/processed/train.jsonl")
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--out", default="outputs/lora-r16")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=0, help="default: 2 * rank")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", choices=TARGETS, default="all")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--qlora", action="store_true", help="load base model in 4-bit")
    ap.add_argument("--full-loss", action="store_true",
                    help="compute loss on prompt+completion instead of completion only")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Native bf16 needs compute capability >= 8 (Ampere+). torch.cuda.is_bf16_supported()
    # also returns True on a T4 because it counts software emulation, which is very slow.
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16 if torch.cuda.is_available() \
        else torch.float32

    # ---------------- data: keep only what the trainer needs ---------------- #
    ds = load_dataset("json", data_files={"train": args.train, "val": args.val})
    ds = ds.select_columns(["prompt", "completion"])
    # Conversational prompt-completion format: TRL applies the chat template and
    # (by default) masks the prompt so loss is computed on the JSON answer only.

    # ---------------- model ---------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",          # NormalFloat4: suits normally-distributed weights
            bnb_4bit_use_double_quant=True,     # also quantise the quantisation constants
            bnb_4bit_compute_dtype=dtype,       # matmuls run in 16-bit
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, quantization_config=quant,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # ---------------- LoRA ---------------- #
    # W' = W + (alpha / r) * B @ A, with A: r x d_in, B: d_out x r, B initialised to 0
    # -> at step 0 the model is exactly the base model.
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha or 2 * args.rank,
        lora_dropout=args.dropout,
        target_modules=TARGETS[args.targets],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---------------- training config ---------------- #
    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,   # effective batch = 4 * 4 = 16
        learning_rate=args.lr,                         # LoRA wants ~10x a full-FT LR
        lr_scheduler_type="cosine",
        warmup_steps=0.05,                             # float in [0,1) = ratio of total steps
        max_length=args.max_length,
        completion_only_loss=not args.full_loss,
        gradient_checkpointing=True,                   # trade compute for activation memory
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    # QLoRA + fp16 (T4): TRL casts the trainable LoRA params of a 4-bit model to bf16, but the
    # fp16 GradScaler cannot unscale bf16 gradients. Keep them in fp32 (the optimizer is only
    # created inside trainer.train(), so it sees the fp32 params).
    if args.qlora and not use_bf16:
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

    trainer.model.print_trainable_parameters()

    trainer.train()
    trainer.save_model(args.out)          # writes adapter_config.json + adapter_model.safetensors
    tokenizer.save_pretrained(args.out)

    with open(f"{args.out}/train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
