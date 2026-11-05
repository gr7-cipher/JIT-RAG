#!/usr/bin/env bash
# Download BEIR collections into experiments/data/. Usage: bash scripts/download_data.sh scifact fiqa ...
set -euo pipefail
cd "$(dirname "$0")/../experiments"; mkdir -p data
for d in "$@"; do
  curl -fL -o "data/$d.zip" "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$d.zip"
  (cd data && unzip -q -o "$d.zip" && rm "$d.zip")
done
