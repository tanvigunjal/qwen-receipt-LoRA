#!/usr/bin/env bash
# Full evaluation matrix. Start the server first:  bash scripts/serve.sh
set -euo pipefail
BASE=${BASE:-Qwen/Qwen3-4B-Instruct-2507}

# 1) base model: free text vs. JSON-schema constrained decoding
python src/eval.py --model "$BASE" --tag base
python src/eval.py --model "$BASE" --tag base-json --structured

# 2) every adapter the server knows about, with and without constrained decoding
for name in $(curl -s localhost:8000/v1/models | python -c \
    "import sys,json;print(' '.join(m['id'] for m in json.load(sys.stdin)['data'] if m.get('parent')))"); do
  python src/eval.py --model "$name" --tag "$name"
  python src/eval.py --model "$name" --tag "$name-json" --structured
done

# 3) serving performance: base alone, one adapter, two adapters mixed in the same batches
python src/bench.py --models "$BASE"
python src/bench.py --models sroie-r16
python src/bench.py --models sroie-r8,sroie-r32

python src/report.py
