import os, sys, pickle, numpy as np, pytrec_eval
R="results/e16"; DS="scifact,nfcorpus,arguana,fiqa,scidocs,trec-covid,quora".split(",")
def load(n): return pickle.load(open(f"{R}/dev_{n}.pkl","rb"))
def nrm(x,h="max"):
    x=np.asarray(x,float)
    if len(x)==0: return x
    return x/x.max() if x.max()>0 else np.zeros_like(x)
def ev(run,qrels):
    e=pytrec_eval.RelevanceEvaluator(qrels,{"ndcg_cut.10","recall.100"}); r=e.evaluate(run)
    return np.mean([r[q]["ndcg_cut_10"] for q in qrels]),np.mean([r[q]["recall_100"] for q in qrels]),{q:r[q]["ndcg_cut_10"] for q in qrels}
def cand(r,kind):
    if kind=="af": return r["in_af"]|(r["rd"]<51)
    if kind=="h50": return (r["rb"]<51)|(r["rd"]<51)
    if kind=="h100": return (r["rb"]<101)|(r["rd"]<101)
    if kind=="af100": return r["in_af"]|(r["rd"]<101)
    if kind=="af200": return r["in_af"]|(r["rd"]<201)
def score(r,m,a,g,afl=True):
    c=r["cos"][m]; b=nrm(r["s0"][m]); l=nrm(r["l"][m]); lex=(1-g)*b+g*l
    return a*nrm(c)+(1-a)*lex
def run(D,kind,a,g):
    o={}
    for r in D["rows"]:
        m=cand(r,kind); ids=np.array(r["ids"])[m]; aa=a(r) if callable(a) else a
        f=score(r,m,aa,g); o[r["q"]]={d:float(s) for d,s in zip(ids,f) if d!=r["q"]} or {"x":0.0}
    return o
