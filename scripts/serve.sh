#!/usr/bin/env bash
# Serve the base model plus LoRA adapters from ONE vLLM server.
# Each adapter is addressable by name in the OpenAI "model" field.
#
#   bash scripts/serve.sh                      # base + adapters found in outputs/
#   ADAPTERS="sroie-r16=outputs/lora-r16" bash scripts/serve.sh
set -euo pipefail

BASE=${BASE:-Qwen/Qwen3-4B-Instruct-2507}
ADAPTERS=${ADAPTERS:-"sroie-r8=outputs/lora-r8 sroie-r16=outputs/lora-r16 sroie-r32=outputs/lora-r32"}

# keep only adapters that exist on disk
MODULES=()
for kv in $ADAPTERS; do
  [[ -d "${kv#*=}" ]] && MODULES+=("$kv") || echo "skip ${kv%%=*} (not trained yet)"
done

LORA_ARGS=()
if [[ ${#MODULES[@]} -gt 0 ]]; then
  LORA_ARGS=(--enable-lora
             --lora-modules "${MODULES[@]}"
             --max-lora-rank 32          # must be >= the largest adapter rank you serve
             --max-loras 4               # adapters resident on GPU at the same time (per batch)
             --max-cpu-loras 8)          # adapters cached in CPU RAM for fast swap-in
fi

exec vllm serve "$BASE" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  "${LORA_ARGS[@]}" \
  ${EXTRA_ARGS:-}
# On a T4 (no bf16) add:  EXTRA_ARGS="--dtype half"
