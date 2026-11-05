#!/bin/zsh
cd "$(dirname "$0")/../experiments"
PY=".venv/bin/python -I"
$=PY e10_final.py lag > logs/e10_lag.log 2>&1
$=PY e10_final.py cascade > logs/e10_cascade.log 2>&1
$=PY e10_final.py latency > logs/e10_latency.log 2>&1
touch logs/E10_DONE
