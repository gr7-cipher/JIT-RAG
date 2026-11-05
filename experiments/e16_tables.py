import os, sys, json; sys.path.insert(0,'.')
from common import *
S=["webis-touche2020","cqa-webmasters","cqa-android","cqa-wordpress","fiqa_fresh","nfcorpus_fresh"]; L=["Touche-2020","CQA Webmasters","CQA Android","CQA WordPress","FiQA (fresh)","NFCorpus (fresh)"]
P={s:load_json("e16",f"test_{s}.json") for s in S}; N={s:len(P[s]["rrf"]["ndcg"]) for s in S}
g=lambda s,k,m="ndcg":mean(P[s][k][m])
def mb(a,b,m="ndcg",nb=4000):
    r=np.random.default_rng(SEED); x=[np.array([P[s][a][m][q]-P[s][b][m][q] for q in P[s][a][m]]) for s in S]
    bs=np.mean([[v[r.integers(0,len(v),len(v))].mean() for _ in range(nb)] for v in x],0); return float(np.mean([v.mean() for v in x])),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
K=["af_f0.0","afr_f0.0","h100_f0.0","rrf","dense"]; o=[]
for s,l in zip(S,L):
    v=[g(s,k) for k in K]; b=max(round(x,3) for x in v); c=" & ".join((f"\\textbf{{{x:.3f}}}" if round(x,3)==b else f"{x:.3f}") for x in v)
    o.append(f"{l} & {N[s]} & {c} & {g(s,'af_f0.1'):.3f} & {g(s,'h100_f0.1'):.3f} & {g(s,'af_f0.5'):.3f} & {g(s,'h100_f0.5'):.3f} \\\\")
mac=lambda k,m="ndcg":float(np.mean([g(s,k,m) for s in S]))
v=[mac(k) for k in K]; b=max(round(x,3) for x in v); o+=["\\midrule",f"Macro nDCG@10 & & "+" & ".join((f"\\textbf{{{x:.3f}}}" if round(x,3)==b else f"{x:.3f}") for x in v)+f" & {mac('af_f0.1'):.3f} & {mac('h100_f0.1'):.3f} & {mac('af_f0.5'):.3f} & {mac('h100_f0.5'):.3f} \\\\"]
v=[mac(k,'recall100') for k in K]; o+=["Macro Recall@100 & & "+" & ".join(f"{x:.3f}" for x in v)+f" & {mac('af_f0.1','recall100'):.3f} & {mac('h100_f0.1','recall100'):.3f} & {mac('af_f0.5','recall100'):.3f} & {mac('h100_f0.5','recall100'):.3f} \\\\"]
open("../resubmission/tables/af_external.tex","w").write("\n".join(o)+"\n"); print("\n".join(o))
for a,b,m in [("af_f0.0","h100_f0.0","ndcg"),("af_f0.0","h100_f0.0","recall100"),("af_f0.0","h50_f0.0","recall100"),("af_f0.0","rrf","ndcg"),("af_f0.0","dense","ndcg"),("af_f0.1","h100_f0.1","ndcg"),("af_f0.5","h100_f0.5","ndcg"),("af_f0.1","h100_f0.1","recall100"),("afr_f0.0","af_f0.0","ndcg"),("afr_f0.0","h100_f0.0","ndcg")]:
    print(a,b,m,"%+.4f [%+.4f,%+.4f]"%mb(a,b,m))
for a,b in [("af_f0.0","rrf"),("af_f0.0","dense")]:
    T=[paired_test(P[s][a]["ndcg"],P[s][b]["ndcg"]) for s in S]; H=holm([t["p"] for t in T]); print(a,b,[f"{s[:8]} {t['diff']:+.3f} p={h:.2g}" for s,t,h in zip(S,T,H)])
