#!/usr/bin/env bash
# Experiments added for the current version of the paper. They need the embeddings and BM25 candidates
# from e01_base.py and e08_compare.py, and the frozen settings stored in results/.
set -uo pipefail
cd "$(dirname "$0")/../experiments"; PY=${PY:-python}
$PY e17_lazy.py                                       # ablation, lazy fusion, backlog, budget, write-back
$PY e19_overload.py                                   # overload replay
$PY e18_live.py grid 0; $PY e18_live.py grid 1        # live load; run alone, about one hour per repeat
$PY e22_idle.py                                       # latency without contention; run alone
$PY e24_generation.py Qwen/Qwen2.5-1.5B-Instruct 150  # local generation check
$PY e20_stats.py; $PY e21_cost.py; $PY e23_pareto.py; $PY e25_shared_check.py; $PY e21_tables.py
