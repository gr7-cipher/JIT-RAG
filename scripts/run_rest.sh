#!/bin/zsh
# Runs every experiment after e01, one at a time, so latency numbers are not disturbed by other jobs.
cd "$(dirname "$0")/../experiments"
PY=".venv/bin/python -I"
run() { echo "=== $* $(date +%H:%M)"; "$@" 2>&1 | grep --line-buffered -vE "Loading weights|LOAD REPORT|UNEXPECTED|^Key|^---|^$|^Notes|position_ids"; }
run $=PY e02b_latency_resume.py > logs/e02b.log
run $=PY e05_stream_erasure_filter.py stream > logs/e05_stream.log
run $=PY e05_stream_erasure_filter.py erasure > logs/e05_erasure.log
run $=PY e05_stream_erasure_filter.py filter > logs/e05_filter.log
run $=PY e05_stream_erasure_filter.py visibility > logs/e05_vis.log
run $=PY e04_cache_adaptive.py cache > logs/e04_cache.log
run $=PY e04_cache_adaptive.py cachelive > logs/e04_live.log
run $=PY e04_cache_adaptive.py adaptive > logs/e04_adaptive.log
run $=PY e03_rerank_sparse_hparams.py lexgap > logs/e03_lexgap.log
run $=PY e03_rerank_sparse_hparams.py ce > logs/e03_ce.log
run $=PY e03_rerank_sparse_hparams.py bm25grid > logs/e03_grid.log
run $=PY e03_rerank_sparse_hparams.py models > logs/e03_models.log
run $=PY e03_rerank_sparse_hparams.py splade > logs/e03_splade.log
echo ALL_DONE $(date +%H:%M) > logs/ALL_DONE
