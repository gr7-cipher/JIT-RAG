#!/usr/bin/env bash
# NOTE: this script has not been run. No MS MARCO or Natural Questions result appears in the paper.
# Runs the full experiment suite on a CUDA machine, adding MS MARCO and Natural Questions.
# Results go to results_a100/, separate from the Apple M1 results used in the manuscript.
#   Needs: Python 3.10+, about 100 GB of disk and at least 64 GB of RAM (MS MARCO is 8.8 M passages).
#   Usage: bash run_a100.sh            (expect many hours; each stage logs to logs_a100/)
set -uo pipefail
cd "$(dirname "$0")/../experiments"
[ -d .venv_a100 ] || python3 -m venv .venv_a100
.venv_a100/bin/pip install -q torch sentence-transformers transformers bm25s PyStemmer hnswlib tantivy pytrec-eval-terrier scikit-learn scipy matplotlib
PY=.venv_a100/bin/python
export JIT_RES=results_a100
export JIT_DATASETS=scifact,nfcorpus,arguana,fiqa,scidocs,trec-covid,quora,nq,msmarco
mkdir -p "$JIT_RES" logs_a100 data
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > logs_a100/gpu.txt || { echo "No GPU found"; exit 1; }
for d in ${JIT_DATASETS//,/ }; do
  if [ ! -f "data/$d/corpus.jsonl" ]; then
    echo "Downloading $d"
    curl -fL -o "data/$d.zip" "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$d.zip" && (cd data && unzip -q -o "$d.zip" && rm "$d.zip")
  fi
done
step() { name=$1; shift; echo "== $name $(date +%H:%M)"; "$@" > "logs_a100/$name.log" 2>&1 || echo "FAILED: $name (see logs_a100/$name.log)"; }
for d in ${JIT_DATASETS//,/ }; do step "e01_$d" $PY e01_base.py "$d"; done
step latency $PY e02_latency.py
step stream $PY e05_stream_erasure_filter.py stream
step erasure $PY e05_stream_erasure_filter.py erasure
step filter $PY e05_stream_erasure_filter.py filter
step cache $PY e04_cache_adaptive.py cache
step cachelive $PY e04_cache_adaptive.py cachelive
step adaptive $PY e04_cache_adaptive.py adaptive
step lexgap $PY e03_rerank_sparse_hparams.py lexgap
step ce $PY e03_rerank_sparse_hparams.py ce
step bm25grid $PY e03_rerank_sparse_hparams.py bm25grid
step models $PY e03_rerank_sparse_hparams.py models
step splade $PY e03_rerank_sparse_hparams.py splade
for w in canary cacheonce backlog latcal latrepeat controller; do step "$w" $PY e06_extra.py "$w"; done
step e07 $PY e07_improve.py
step e08_enc $PY e08_compare.py encode bge
step e08_minilm $PY e08_compare.py eval minilm
step e08_bge $PY e08_compare.py eval bge
step e08_rep_minilm $PY e08_report.py minilm
step e08_rep_bge $PY e08_report.py bge
for w in lag lagimp cascade; do step "e10_$w" $PY e10_final.py "$w"; done
step e10_report $PY e10_report.py
for w in latency lathyb; do step "e10_$w" $PY e10_final.py "$w"; done
# JIT-AF: settings stay frozen from the M1 development queries, so MS MARCO and NQ are fully held out.
cp results/e13_pick.json results/e14_pick.json "$JIT_RES"/
for w in budget unseen warm overload latency; do step "e13_$w" $PY e13_af.py "$w"; done
for w in lag unseen latency; do step "e14_$w" $PY e14_afh.py "$w"; done
step e14_report $PY e14_report.py
step e15 $PY e15_final_check.py
# Pre-set external check: frozen JIT-AF+ANN against the agreement rule and rivals on the two largest collections.
cp results/e16_pick.json "$JIT_RES"/
JIT_E16_SETS=nq,msmarco step e16_test $PY e16_test.py
step e16_bench $PY e16_bench.py
step tables $PY make_tables.py
JIT_OUT="$JIT_RES/paper" step e10_tables $PY e10_tables.py
JIT_OUT="$JIT_RES/paper" step e14_tables $PY e14_tables.py
echo "Finished. Send back results_a100/ (without the .npy files) and logs_a100/."
