# JIT-RAG experiments

Every number in `resubmission/JIT-RAG-v2.tex` comes from these scripts. Results are in `results/`, logs in `logs/`.

| Script | What it produces |
|---|---|
| `e01_base.py <collection>` | Quality of all systems, first-stage recall, embeddings, HNSW timings |
| `e02_latency.py`, `e02b_latency_resume.py` | Latency by K, model, truncation, batch size and device |
| `e03_rerank_sparse_hparams.py ce\|splade\|models\|bm25grid\|lexgap` | Cross-encoder, SPLADE, model sweep, BM25 grid, lexical gap |
| `e04_cache_adaptive.py cache\|cachelive\|adaptive` | Cache replay, live cache, predicted K |
| `e05_stream_erasure_filter.py stream\|visibility\|erasure\|filter` | Visibility lag, hidden documents, HNSW deletion, filters |
| `e06_extra.py canary\|cacheonce\|backlog\|latcal\|latrepeat\|ksweep\|controller` | Inverted-index deletion, no-repeat cache, hybrid, latency calibration, controller |
| `e07_improve.py`, `e07_report.py` | First fusion trials: RRF, convex and RM3 candidates over the BM25 list |
| `e08_compare.py encode\|eval <minilm\|bge>`, `e08_report.py <enc>` | JIT-CC against dense, RRF and Hybrid CC, with leave-one-domain-out weights |
| `e09_cascade.py`, `e09b_hybrid_cascade.py` | Early cross-encoder trials with MiniLM, superseded by `e10_final.py cascade` |
| `e10_final.py lag\|lagimp\|cascade\|latency\|lathyb` | Final comparison with bge-small: backlog, signal-aware rivals, cross-encoder stage, latency |
| `e10_report.py`, `e10_tables.py` | Paired tests, RQ8 tables, the backlog figure and the utility sweep |
| `e11_robust.py`, `e11_report.py` | Candidate expansion and normalizations at a fixed encoding budget, with a dev/held-out split |
| `e12_refine.py`, `e12_select.py` | Refinement scan, selected on development queries only |
| `e13_af.py pick\|budget\|unseen\|warm\|overload\|latency` | JIT-AF: frozen settings, budget, outside query sets, write-back, overload simulation, live latency |
| `e14_afh.py grid\|lag\|unseen\|latency`, `e14_report.py`, `e14_tables.py` | JIT-AF+ANN, final statistics, RQ9 tables and figure |
| `e15_final_check.py` | Last check on FiQA and NFCorpus train queries, unused until then |
| `e16_dev_cache.py`, `e16_lab.py`, `e16_a.py`, `e16_b.py`, `e16_c.py`, `e16_fit.py` | FiQA diagnosis and the agreement-weighted rule, on development queries only; frozen in `results/e16_pick.json` with its adoption criteria |
| `e16_encode.py`, `e16_test.py`, `e16_report.py`, `e16_tables.py`, `e16_bench.py` | RQ10: external test on Touche-2020, three CQADupStack forums and fresh FiQA and NFCorpus queries; encoder speed |
| `make_tables.py` | LaTeX table fragments, figures and `results/numbers.json` |

To reproduce on a CUDA machine, including MS MARCO and Natural Questions, run the following. It needs about 100 GB of disk and at least 64 GB of RAM, and it writes to `results_a100/`.

```bash
bash run_a100.sh
```

BEIR collections go in `data/<name>/`, downloaded from https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/.
