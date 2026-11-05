"""Shared helpers for the JIT-RAG experiments: data loading, BM25 scoring, evaluation, statistics."""
import json, os, random, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, os.environ.get("JIT_RES", "results"))
FIG = os.path.join(ROOT, "figures")
DATASETS = os.environ.get("JIT_DATASETS", "scifact,nfcorpus,arguana,fiqa,scidocs,trec-covid,quora").split(",")
K_GRID = [10, 20, 50, 100, 200, 500, 1000, 2000]
MAX_Q = 500
SEED = 13
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def res_dir(name):
    p = os.path.join(RES, name)
    os.makedirs(p, exist_ok=True)
    return p


def save_json(obj, *path):
    p = os.path.join(RES, *path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f)


def load_json(*path):
    with open(os.path.join(RES, *path)) as f:
        return json.load(f)


def load_dataset(name, max_q=MAX_Q, split="test"):
    """Load a BEIR dataset. title+text is the document string. Queries are the judged test
    queries, subsampled to max_q with a fixed seed so that every system sees the same set."""
    base = os.path.join(DATA, name)
    doc_ids, docs = [], []
    with open(os.path.join(base, "corpus.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            doc_ids.append(str(o["_id"]))
            docs.append(((o.get("title") or "") + " " + (o.get("text") or "")).strip())
    q_all = {}
    with open(os.path.join(base, "queries.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            q_all[str(o["_id"])] = o["text"]
    qrels = {}
    with open(os.path.join(base, "qrels", split + ".tsv")) as f:
        next(f)
        for line in f:
            q, d, r = line.rstrip("\n").split("\t")
            qrels.setdefault(q, {})[d] = int(float(r))
    # A query needs at least one positive judgment that is not the query's own document.
    qids = sorted(q for q in qrels if q in q_all and any(r > 0 and d != q for d, r in qrels[q].items()))
    if max_q and len(qids) > max_q:
        rng = random.Random(SEED)
        qids = sorted(rng.sample(qids, max_q))
    queries = [q_all[q] for q in qids]
    qrels = {q: {d: r for d, r in qrels[q].items() if d != q} for q in qids}
    return doc_ids, docs, qids, queries, qrels


class BM25:
    """Lucene-variant BM25 built by bm25s, scored here with a weighted accumulator so that
    query-side expansion (RM3) and document filters use one code path."""

    def __init__(self, docs, k1=1.2, b=0.75):
        import bm25s, Stemmer
        self.stemmer = Stemmer.Stemmer("english")
        self.tok = bm25s.tokenize(docs, stopwords="en", stemmer=self.stemmer, show_progress=False)
        self.ret = bm25s.BM25(method="lucene", k1=k1, b=b)
        self.ret.index(self.tok, show_progress=False)
        s = self.ret.scores
        self.data, self.indices, self.indptr = s["data"], s["indices"], s["indptr"]
        self.n = int(s["num_docs"])
        self.vocab = self.ret.vocab_dict

    def nbytes(self):
        return int(self.data.nbytes + self.indices.nbytes + self.indptr.nbytes)

    def q_terms(self, text):
        import bm25s
        toks = bm25s.tokenize([text], stopwords="en", stemmer=self.stemmer, return_ids=False, show_progress=False)[0]
        return [self.vocab[t] for t in toks if t in self.vocab]

    def score(self, weights, mask=None):
        s = np.zeros(self.n, dtype=np.float32)
        for t, w in weights.items():
            a, b = self.indptr[t], self.indptr[t + 1]
            s[self.indices[a:b]] += w * self.data[a:b]
        if mask is not None:
            s *= mask
        return s

    def topk(self, weights, k, mask=None):
        s = self.score(weights, mask)
        nz = np.count_nonzero(s)
        k = min(k, nz)
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return top, s[top]

    def search(self, text, k, mask=None):
        w = {}
        for t in self.q_terms(text):
            w[t] = w.get(t, 0.0) + 1.0
        return self.topk(w, k, mask)

    def rm3_weights(self, text, fb_docs=10, fb_terms=10, lam=0.5):
        """RM3 using only the query and the inverted index's forward token lists: no new state."""
        w = {}
        for t in self.q_terms(text):
            w[t] = w.get(t, 0.0) + 1.0
        if not w:
            return w
        top, sc = self.topk(w, fb_docs)
        if len(top) == 0:
            return w
        sc = sc / sc.sum()
        rel = {}
        ids = self.tok.ids
        for d, p in zip(top, sc):
            toks = ids[int(d)]
            if not len(toks):
                continue
            u, c = np.unique(np.asarray(toks), return_counts=True)
            for t, n in zip(u, c):
                rel[int(t)] = rel.get(int(t), 0.0) + float(p) * n / len(toks)
        keep = sorted(rel.items(), key=lambda x: -x[1])[:fb_terms]
        tot = sum(v for _, v in keep) or 1.0
        qt = sum(w.values())
        out = {t: lam * v / qt for t, v in w.items()}
        for t, v in keep:
            out[t] = out.get(t, 0.0) + (1 - lam) * v / tot
        return out


def encoder(model=BASE_MODEL, max_seq=256, dev=None):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model, device=dev or device())
    m.max_seq_length = max_seq
    return m


def encode(m, texts, bs=64, progress=False):
    if len(texts) == 0:
        return np.zeros((0, m.get_sentence_embedding_dimension()), dtype=np.float32)
    return m.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True,
                    show_progress_bar=progress).astype(np.float32)


def run_from_ranked(doc_ids, qid, idx, scores, depth=1000):
    """idx/scores in rank order. Drops the query's own document, as BEIR's evaluator does."""
    out = {}
    for i, s in zip(idx, scores):
        d = doc_ids[int(i)]
        if d == qid:
            continue
        out[d] = float(s)
        if len(out) >= depth:
            break
    return out


def evaluate(run, qrels):
    """Per-query metrics via trec_eval (pytrec_eval): nDCG@10, Recall@100, MRR@10."""
    import pytrec_eval
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.100"})
    r = ev.evaluate({q: (run.get(q) or {"__none__": 0.0}) for q in qrels})
    out = {"ndcg": {}, "recall100": {}, "mrr": {}}
    for q in qrels:
        out["ndcg"][q] = r[q]["ndcg_cut_10"]
        out["recall100"][q] = r[q]["recall_100"]
        ranked = sorted((run.get(q) or {}).items(), key=lambda x: (-x[1], x[0]))[:10]
        rr = 0.0
        for i, (d, _) in enumerate(ranked):
            if qrels[q].get(d, 0) > 0:
                rr = 1.0 / (i + 1)
                break
        out["mrr"][q] = rr
    return out


def mean(d):
    return float(np.mean(list(d.values()))) if d else float("nan")


def paired_test(a, b, n_boot=2000):
    """a, b: {qid: value}. Mean difference a-b, 95% bootstrap CI, Wilcoxon signed-rank p."""
    from scipy.stats import wilcoxon
    qs = sorted(set(a) & set(b))
    x = np.array([a[q] - b[q] for q in qs])
    rng = np.random.default_rng(SEED)
    boots = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(1)
    p = 1.0 if np.allclose(x, 0) else float(wilcoxon(x, zero_method="wilcox").pvalue)
    return {"diff": float(x.mean()), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)), "p": p, "n": len(x)}


def holm(ps):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = np.argsort(ps)
    adj = [0.0] * len(ps)
    cur = 0.0
    for r, i in enumerate(order):
        cur = max(cur, (len(ps) - r) * ps[i])
        adj[i] = min(1.0, cur)
    return adj


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
    sys.stdout.flush()
