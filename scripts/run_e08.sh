#!/bin/zsh
cd "$(dirname "$0")/../experiments"; PY=".venv/bin/python -I"; rm -f logs/E08_DONE
$=PY e08_compare.py eval minilm > logs/e08_minilm.log 2>&1
$=PY e09_cascade.py > logs/e09.log 2>&1
touch logs/E08_FIRST
$=PY e08_compare.py encode bge > logs/e08_bge_enc.log 2>&1
$=PY e08_compare.py eval bge > logs/e08_bge.log 2>&1
touch logs/E08_DONE
