"""E24: a small end-to-end check with a local model and no external service. Retrieval runs at a 50% backlog on FiQA held-out queries.
The model answers from the top 3 passages. Answers are scored against the best judged relevant passage, by token F1 and by bge-small cosine.
These are weak proxies for answer quality, and a small model limits what they can show."""
import os, sys, json, re, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a = sys.argv[1:]; sys.argv = ["x", "noop"]
from e17_lazy import *
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL = _a[0] if _a else "Qwen/Qwen2.5-1.5B-Instruct"; NAME = "fiqa"; NQ = int(_a[1]) if len(_a) > 1 else 150; F_ = 0.5; TOPN = 3
doc_ids, docs, qids, queries, qrels = load_dataset(NAME, max_q=500); E = np.asarray(np.load(res(f"{NAME}/bge_emb.npy"))); Q = np.load(res(f"{NAME}/bge_qemb.npy")); C = np.load(res(f"{NAME}/bm25_cands.npy"), allow_pickle=True); bm = BM25(docs); idf = np.log(1 + bm.n / (1 + np.diff(bm.indptr))); n = len(E); get = lambda ids, v: E[ids] @ v
h = np.random.default_rng(SEED).random(n) < F_; rs = np.random.default_rng(SEED).permutation(len(qids)); idx = sorted(rs[len(qids) // 2:].tolist())[:NQ]
tok = AutoTokenizer.from_pretrained(MODEL); lm = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(device()); lm.eval(); enc = encoder(BGE, 256); pos = {d: i for i, d in enumerate(doc_ids)}; cache = {}
def gen(q, pas):
    key = (q, tuple(pas))
    if key in cache: return cache[key]
    ctx = "\n\n".join(f"[{i + 1}] {' '.join(docs[p].split()[:150])}" for i, p in enumerate(pas)); msg = [{"role": "system", "content": "Answer the question using only the passages. Be brief."}, {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {q}"}]
    ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(lm.device)
    with torch.no_grad(): o = lm.generate(**ids, max_new_tokens=80, do_sample=False)
    a = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip(); cache[key] = a; return a
def toks(x): return re.findall(r"[a-z0-9]+", x.lower())
def f1(a, b):
    A, B = toks(a), toks(b); c = {}; 
    for t in B: c[t] = c.get(t, 0) + 1
    m = 0
    for t in A:
        if c.get(t, 0) > 0: m += 1; c[t] -= 1
    return 0.0 if not m else 2 * m / (len(A) + len(B))
S = {k: {"f1": [], "cos": []} for k in ["full_index", "sa100", "lazy50", "afh"]}; answers = []; t0 = time.time()
for j, i in enumerate(idx):
    q = qids[i]; v = Q[i]; c0 = np.asarray(C[i], dtype=np.int64); w0 = weights(bm, queries[i]); S0 = bm.score(w0) if w0 else np.zeros(n, np.float32); sim = E @ v; order = np.argsort(-sim)[:3000]; gold = [pos[d] for d, r in qrels[q].items() if r > 0 and d in pos]
    if not gold: continue
    top0 = order[:101]; top = visible(order, h, 101); R = {}
    us, f, _ = hyb(c0, S0, sim, top0, 100, None); R["full_index"] = us[np.argsort(-f, kind="stable")[:TOPN]]
    us, f, _ = hyb(c0, S0, sim, top, 100, h); R["sa100"] = us[np.argsort(-f, kind="stable")[:TOPN]]
    us, f, _ = hyb(c0, S0, sim, top, 50, h, lazy=True); R["lazy50"] = us[np.argsort(-f, kind="stable")[:TOPN]]
    u0, cs0, l0, _ = af_run(bm, idf, w0, S0, c0, v, get); ex = ann_extra(order, sim, h, u0); ua = np.concatenate([u0, ex]); ca = np.concatenate([cs0, sim[ex]]); f = af_score(ua, ca, S0, l0, NR, AF_A, AF_G); R["afh"] = ua[np.argsort(-f, kind="stable")[:TOPN]]
    gv = E[gold]; row = {"q": q}
    for k, pas in R.items():
        a = gen(queries[i], [int(x) for x in pas]); S[k]["f1"].append(max(f1(a, docs[g]) for g in gold)); av = encode(enc, [a if a else " "])[0]; S[k]["cos"].append(float((gv @ av).max())); row[k] = a
    answers.append(row)
    if j % 10 == 0: log(j, time.time() - t0, {k: round(float(np.mean(x["f1"])), 3) for k, x in S.items()}); json.dump({"model": MODEL, "n": len(answers), "scores": S}, open(res("e24_generation.json"), "w"))
json.dump({"model": MODEL, "n": len(answers), "scores": S, "answers": answers}, open(res("e24_generation.json"), "w"))
for k in S:
    print(k, "F1 %.4f cos %.4f" % (np.mean(S[k]["f1"]), np.mean(S[k]["cos"])))
rng = np.random.default_rng(SEED)
for a, b in [("lazy50", "sa100"), ("afh", "lazy50"), ("full_index", "lazy50")]:
    for m in ["f1", "cos"]:
        x = np.array(S[a][m]) - np.array(S[b][m]); bs = rng.choice(x, size=(4000, len(x))).mean(1); print(a, b, m, "%+.4f [%+.4f,%+.4f]" % (x.mean(), *np.percentile(bs, [2.5, 97.5])))
