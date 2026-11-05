# Query-time embedding in retrieval-augmented generation

Code, per-query results and logs for the paper "Query-Time Embedding in Retrieval-Augmented Generation: Absorbing Encoder Lag in Indexed Fusion Retrieval". The paper is under review, so its text is not in this repository.

When writes exceed encoder capacity, documents reach the inverted index but wait for a vector, and dense search cannot find them. Two designs here deal with that by encoding candidates at query time and storing the vectors:

- Lazy fusion is convex fusion of BM25 and an ANN index, plus query-time encoding of waiting documents among the BM25 top K.
- JIT-AF+ANN (just-in-time adaptive fusion) also expands candidates through rounds of pseudo-relevance feedback. JIT-AF is the same design without an ANN index.

## When this helps and when it does not

Query-time encoding helps during encoder overload, when quality matters more than latency. With 10% of the corpus waiting, encoding about five candidates per query restored the quality of a complete index.

It does not help when the encoder keeps up, since nothing is waiting. It does not shrink the backlog on a shared accelerator, because query-time encodings replace background encodings almost one for one. It costs latency and throughput, and the query path needs encoder capacity. Encoding at ingest stays the default.

The feedback loop in JIT-AF+ANN added a little Recall@100 and no nDCG@10 over fusion alone, and it saturated first under live load.

## Results

All values come from one Apple M1 laptop with 8 GB of memory and bge-small-en-v1.5. Quality is macro nDCG@10 on held-out queries from seven BEIR collections.

| Waiting | Lazy fusion, K=50 | Encodings per query | JIT-AF+ANN | Encodings per query | Indexed fusion, signal-aware |
|---|---|---|---|---|---|
| 0% | 0.5879 | 0 | 0.5869 | 0 | 0.5879 |
| 10% | 0.5876 | 4.9 | 0.5867 | 9.8 | 0.5692 |
| 50% | 0.5838 | 24.4 | 0.5848 | 49.3 | 0.4912 |

Live test on FiQA with one shared accelerator, 100 writes and 5 queries per second, two runs each:

| System | Median latency (ms) | p95 latency (ms) | Final backlog share |
|---|---|---|---|
| Indexed fusion, signal-aware | 91, 102 | 148, 185 | 0.160, 0.176 |
| Lazy fusion, K=50 | 467, 419 | 806, 1,070 | 0.165, 0.164 |
| JIT-AF+ANN | 903, 1,483 | 1,826, 7,416 | 0.169, 0.174 |

A shared-capacity model predicted the backlog of the overloaded runs with a mean absolute error of 0.004. A model that treats query-time encoding as added capacity had an error of 0.065.

## Repository layout

```
src/jit_rag/    reference implementation of lazy fusion, JIT-AF and the write-back store
tests/          unit tests, including at-most-once encoding and backlog invariance
examples/       a minimal runnable example
experiments/    numbered experiment scripts, results/ (per-query JSON) and logs/
scripts/        downloads, run scripts and the table rebuild
generated/      tables and figures rebuilt from stored results
```

The experiments use their own vectorised code paths and produced every number in the paper. `src/jit_rag` exists so the algorithms can be read and tested separately. Embeddings, BEIR data and pickled caches are not stored here.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on macOS with Metal. A fresh clone passed the unit tests, rebuilt every table from stored results, and reproduced the stored SciFact results exactly with `e01_base.py`, `e08_compare.py encode bge` and `e17_lazy.py`. Run the unit tests with:

```bash
PYTHONPATH=src pytest -q tests
```

## Minimal usage

```bash
PYTHONPATH=src python examples/minimal.py
```

Output on our machine:

```
encoded at query time: 3
1 Mortgage payments rise when lending rates go up.
0 The central bank raised interest rates by a quarter point.
3 A savings account pays interest every month.
after a repeat query, total encodings: 3
```

## Data

BEIR collections are public. This downloads them to `experiments/data/`:

```bash
bash scripts/download_data.sh scifact nfcorpus arguana fiqa scidocs trec-covid quora
```

External sets are `webis-touche2020`, `cqa-webmasters`, `cqa-android` and `cqa-wordpress`.

## Reproducing the paper

Rebuild statistics, tables and figures from stored results, with no data or GPU:

```bash
python scripts/rebuild_tables.py
```

To run the experiments, first produce embeddings and candidates with `e01_base.py <collection>` and `e08_compare.py`, then use `scripts/run_paper_v3.sh`. Run `e18_live.py` and `e22_idle.py` with nothing else on the machine. `experiments/README_scripts.md` describes the earlier scripts. Older scripts keep the labels from when they were written.

| In the paper | Script | Stored result |
|---|---|---|
| Table 1, collections | `e01_base.py`, `e21_tables.py` | `results/<collection>/meta.json`, `results/e20_stats.json` |
| Tables 2 and 3, Figure 2, backlog and tests | `e17_lazy.py`, `e20_stats.py` | `results/e17/`, `results/e20_stats.json`, `results/e17_margin.json` |
| Table 4, ablation | `e17_lazy.py`, `e20_stats.py` | same |
| Table 5, complete index | `e17_lazy.py`, `e20_stats.py` | same |
| Tables 6 and 7, budget and write-back | `e17_lazy.py`, `e20_stats.py` | same |
| Table 8, Figure 3, live load | `e18_live.py`, `e25_shared_check.py`, `e23_pareto.py` | `results/e18_live.json`, `results/e25_shared.json`, `results/e23_pareto.json` |
| Table 9, latency without contention | `e22_idle.py`, `e14_afh.py latency`, `e13_af.py latency`, `e10_final.py` | `results/e22_idle.json`, `results/e14_latency.json`, `results/e13_latency.json`, `results/e10_latency_hyb.json`, `results/e10_latency.json` |
| Table 10, overload replay | `e19_overload.py` | `results/e19_overload.json` |
| Table 11, cost | `e21_cost.py` | `results/e21_cost.json` |
| Figure 4, freshness | `e05_stream_erasure_filter.py stream` | `results/stream.json` |
| Earlier designs and supporting measurements | `e01` to `e16` | `results/numbers.json` and others |
| Section 6.9, generation check | `e24_generation.py` | `results/e24_generation.json` |

Quality scripts read stored vectors and count encodings, so they do not need a GPU once embeddings exist.

## Limitations

These results cover retrieval on one laptop. The only generation test is a small check with a local 1.5B model and proxy scores, which found no measurable difference in answers. MS MARCO and Natural Questions were not run: `scripts/run_a100.sh` is included and has not been run. The largest corpus has 523,000 documents. Backlogs in the quality experiments are uniform masks on collections with no timestamps in the BEIR files, and real backlogs are recent and clustered. The live test uses one collection, Python threads, and in-memory text and vectors, with memory pressure on the machine. JIT-AF+ANN received more tuning than its rivals. Prices in the cost model are assumptions.

## Citation and licence

See `CITATION.cff`. The paper has no DOI yet. Code is under the Apache License 2.0 in `LICENSE`. BEIR collections have their own licences.
