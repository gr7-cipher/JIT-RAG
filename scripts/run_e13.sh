#!/bin/zsh
cd "$(dirname "$0")/../experiments"
PY=".venv/bin/python -I"
for w in budget unseen warm overload latency; do $=PY e13_af.py $w > logs/e13_$w.log 2>&1; done
touch logs/E13_DONE
