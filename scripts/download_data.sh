#!/usr/bin/env bash
# Download BEIR collections into experiments/data/.
# Usage: bash scripts/download_data.sh scifact fiqa cqa-android ...
# cqa-<forum> names are extracted from the single BEIR cqadupstack archive.
set -euo pipefail
cd "$(dirname "$0")/../experiments"; mkdir -p data
BASE="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
for d in "$@"; do
  if [ -d "data/$d" ]; then echo "data/$d exists, skipping"; continue; fi
  case "$d" in
    cqa-*)
      forum="${d#cqa-}"
      if [ ! -d data/cqadupstack ]; then
        curl -fL -o data/cqadupstack.zip "$BASE/cqadupstack.zip"
        (cd data && unzip -q -o cqadupstack.zip && rm cqadupstack.zip)
      fi
      cp -R "data/cqadupstack/$forum" "data/$d" ;;
    *)
      curl -fL -o "data/$d.zip" "$BASE/$d.zip"
      (cd data && unzip -q -o "$d.zip" && rm "$d.zip") ;;
  esac
done
