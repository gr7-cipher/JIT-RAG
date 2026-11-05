"""E16 step 2: fit and freeze the agreement-weighted rule on DEV-half queries only, before any new data is opened."""
import sys, json; sys.path.insert(0,'.')
from e16_lab import *
from scipy.stats import rankdata
def rho(c,s):
    if len(c)<3 or np.ptp(s)==0 or np.ptp(c)==0: return 0.0
    return float(np.corrcoef(rankdata(c),rankdata(s))[0,1])
ALL={n:load(n) for n in DS}
def prep(kind,g):
    P={}
    for n in DS:
        L=[]
        for r in ALL[n]["rows"]:
            m=cand(r,kind); L.append((r["q"],np.array(r["ids"])[m],nrm(r["cos"][m]),(1-g)*nrm(r["s0"][m])+g*nrm(r["l"][m]),rho(r["cos"][m],r["s0"][m])))
        P[n]=L
    return P
def macro(P,a0,b,mu,sd):
    o=[]; per={}
    for n in DS:
        run={}
        for q,ids,c,lx,rh in P[n]:
            a=min(1,max(.7,a0+b*(rh-mu)/sd)); f=a*c+(1-a)*lx; run[q]={d:float(s) for d,s in zip(ids,f) if d!=q} or {"x":0.0}
        v=ev(run,ALL[n]["qrels"])[0]; o.append(v); per[n]=round(v,4)
    return float(np.mean(o)),per
out={}
for tag,kind,g in [("af","af",.3),("h100","h100",0.0),("h100p","h100",.3)]:
    P=prep(kind,g); v=[x[4] for n in DS for x in P[n]]; mu,sd=float(np.mean(v)),float(np.std(v)); best=None
    for a0 in [.8,.825,.85,.875,.9]:
        for b in [0,-.01,-.02,-.03,-.04,-.05,-.075,-.1]:
            s,per=macro(P,a0,b,mu,sd)
            if best is None or s>best[0]: best=(s,a0,b,per)
    out[tag]={"mu":mu,"sd":sd,"a0":best[1],"b":best[2],"g":g,"dev_macro":best[0],"dev":best[3]}; print(tag,out[tag],flush=True)
out["criteria"]={"adopt_if":"on new data the rule's macro nDCG@10 is not below frozen JIT-AF+ANN (paired 95% CI lower bound >= -0.002), its FiQA fresh gap to Hybrid K=100 has a CI containing 0 or positive, and no new collection is significantly worse than frozen JIT-AF+ANN after Holm correction","new_data":["webis-touche2020","cqa-webmasters","cqa-android","cqa-wordpress","fiqa fresh train queries","nfcorpus fresh train queries"]}
json.dump(out,open("results/e16_pick.json","w"),indent=1)
