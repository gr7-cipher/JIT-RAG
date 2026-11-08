#!/bin/zsh
cd "$(dirname "$0")/../experiments"
PY=".venv/bin/python -I"
run() { echo "=== $* $(date +%H:%M)"; "$@" 2>&1 | grep --line-buffered -vE "Loading weights|LOAD REPORT|UNEXPECTED|^Key|^---|^$|^Notes|position_ids|HF_TOKEN"; }
rm -f logs/MORE_DONE
run $=PY e04_cache_adaptive.py adaptive > logs/e04_adaptive.log
run $=PY e03_rerank_sparse_hparams.py lexgap > logs/e03_lexgap.log
run $=PY e05_stream_erasure_filter.py visibility > logs/e05_vis.log
run $=PY e06_extra.py canary > logs/e06_canary.log
run $=PY e06_extra.py cacheonce > logs/e06_cacheonce.log
run $=PY e06_extra.py backlog > logs/e06_backlog.log
run $=PY e06_extra.py latcal > logs/e06_latcal.log
run $=PY e06_extra.py latrepeat > logs/e06_latrepeat.log
run $=PY e06_extra.py controller > logs/e06_controller.log
echo DONE $(date +%H:%M) > logs/MORE_DONE
