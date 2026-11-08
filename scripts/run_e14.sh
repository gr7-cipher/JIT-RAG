#!/bin/zsh
cd "$(dirname "$0")/../experiments"
PY=".venv/bin/python -I"
for w in grid lag unseen latency; do $=PY e14_afh.py $w > logs/e14_$w.log 2>&1; done
touch logs/E14B_DONE
