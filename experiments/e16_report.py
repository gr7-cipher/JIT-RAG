import os, sys, json; sys.path.insert(0,'.')
from common import *
S=["webis-touche2020","cqa-webmasters","cqa-android","cqa-wordpress","fiqa_fresh","nfcorpus_fresh"]; P={s:load_json("e16",f"test_{s}.json") for s in S}
SY=["afr","af","af9","h50","h100","h100r","h100p"]
out={}
for f in ["0.0","0.1","0.5"]:
    print(f"\n== backlog {f}: nDCG@10 / R@100"); print("set".ljust(18)+"".join(k.ljust(14) for k in SY+(["rrf","dense"] if f=="0.0" else [])))
    ks=[k+"_f"+f for k in SY]+(["rrf","dense"] if f=="0.0" else [])
    for s in S: print(s[:17].ljust(18)+"".join(f"{mean(P[s][k]['ndcg']):.3f}/{mean(P[s][k]['recall100']):.3f}  " for k in ks))
    print("macro".ljust(18)+"".join(f"{np.mean([mean(P[s][k]['ndcg']) for s in S]):.4f}/{np.mean([mean(P[s][k]['recall100']) for s in S]):.3f} " for k in ks))
def mb(a,b,met="ndcg",sets=S,nb=4000):
    r=np.random.default_rng(SEED); d=np.array([[0.0]]); x=[np.array([P[s][a][met][q]-P[s][b][met][q] for q in P[s][a][met]]) for s in sets]
    bs=np.mean([[v[r.integers(0,len(v),len(v))].mean() for _ in range(nb)] for v in x],0); return float(np.mean([v.mean() for v in x])),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))
print("\nmacro paired differences, 95% stratified bootstrap")
for a,b,f in [("afr","af","0.0"),("afr","h100","0.0"),("afr","h50","0.0"),("afr","h100p","0.0"),("afr","h100r","0.0"),("af","h100","0.0"),("af9","h100","0.0"),("afr","af9","0.0"),("afr","h100","0.1"),("afr","h100","0.5"),("afr","h100p","0.1"),("afr","h100p","0.5"),("afr","af","0.1"),("afr","af","0.5")]:
    A,B=a+"_f"+f,b+"_f"+f; n=mb(A,B); rr=mb(A,B,"recall100"); out[f"{a}-{b}@{f}"]={"ndcg":n,"recall":rr}; print(f"{a}-{b} f={f}: nDCG {n[0]:+.4f} [{n[1]:+.4f},{n[2]:+.4f}] R {rr[0]:+.4f} [{rr[1]:+.4f},{rr[2]:+.4f}]")
for a,b in [("afr","rrf"),("afr","dense")]:
    n=mb(a+"_f0.0",b); print(f"{a}-{b}: nDCG {n[0]:+.4f} [{n[1]:+.4f},{n[2]:+.4f}]"); out[f"{a}-{b}"]={"ndcg":n}
print("\nper set: afr vs af (Holm), afr vs h100, af vs h100")
for pair in [("afr","af"),("afr","h100"),("af","h100"),("afr","h100p")]:
    T=[paired_test(P[s][pair[0]+"_f0.0"]["ndcg"],P[s][pair[1]+"_f0.0"]["ndcg"]) for s in S]; H=holm([t["p"] for t in T]); print(pair)
    for s,t,h in zip(S,T,H): print("  ",s[:17].ljust(18),f"{t['diff']:+.4f} [{t['lo']:+.4f},{t['hi']:+.4f}] n={t['n']} holm p={h:.3g}"); out[f"{pair[0]}-{pair[1]}|{s}"]={**t,"holm":h}
save_json(out,"e16_report.json")
