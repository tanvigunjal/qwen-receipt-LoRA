#!/usr/bin/env bash
# Download SROIE (626 labelled receipts, cleaned version) and build the JSONL splits.
set -euo pipefail
mkdir -p data/raw
if [[ ! -d data/raw/ICDAR-2019-SROIE ]]; then
  git clone --depth 1 https://github.com/zzzDavid/ICDAR-2019-SROIE.git data/raw/ICDAR-2019-SROIE
fi
python src/data_prep.py --sroie-dir data/raw/ICDAR-2019-SROIE/data --out-dir data/processed
