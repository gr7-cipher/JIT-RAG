import sys; sys.path.insert(0,'.')
from e16_lab import *
from scipy.stats import spearmanr
ALL={n:load(n) for n in DS}
A=[.7,.75,.8,.85,.9,.95,1.0]; G=[0,.15,.3]
def feats(r):
    m=cand(r,"af"); c=r["cos"][m]; s=r["s0"][m]; l=r["l"][m]; o={}
    td=np.argsort(-c); tb=np.argsort(-s)
    for k in (10,20): o[f"ov{k}"]=len(set(td[:k])&set(tb[:k]))/k
    lx=.7*nrm(s)+.3*nrm(l); tl=np.argsort(-lx); o["ovl10"]=len(set(td[:10])&set(tl[:10]))/10
    o["rho"]=spearmanr(c,s).statistic if len(c)>2 and s.std()>0 else 0.0
    w=1/(1+np.arange(len(td))); t=np.empty(len(td)); t[tb]=np.arange(len(td)); o["wrank"]=float((1/(1+t[td[:10]])).sum()/w[:10].sum())
    ss=np.sort(s)[::-1]; cc=np.sort(c)[::-1]
    o["bgap"]=float(1-ss[min(9,len(ss)-1)]/ss[0]) if ss[0]>0 else 0; o["cgap"]=float(1-cc[min(9,len(cc)-1)]/cc[0]); o["ctop"]=float(cc[0]); o["cmean10"]=float(cc[:10].mean())
    o["nterm"]=r["nterms"]; o["midf"]=float(np.mean(r["idfs"])) if r["idfs"] else 0; o["maxidf"]=float(max(r["idfs"])) if r["idfs"] else 0
    return o
Q={}; F={}
for n in DS:
    D=ALL[n]; per={}
    for a in A:
        for g in G:
            per[(a,g)]=ev(run(D,"af",a,g),D["qrels"])[2]
    Q[n]=per; F[n]={r["q"]:feats(r) for r in D["rows"]}
pickle.dump((Q,F),open(f"{R}/lab_b.pkl","wb"))
print("best per collection (dev):")
tot=[]
for n in DS:
    m={k:np.mean(list(v.values())) for k,v in Q[n].items()}; b=max(m,key=m.get); tot.append(m[b]); print(n[:8],b,round(m[b],4),"af.8.3",round(m[(.8,.3)],4),".9/0",round(m[(.9,0)],4))
print("collection oracle macro",round(np.mean(tot),4))
print("query oracle macro",round(np.mean([np.mean([max(Q[n][k][q] for k in Q[n]) for q in Q[n][(.8,.3)]]) for n in DS]),4))
names=list(next(iter(F[DS[0]].values())))
print("\ncorr(feature, ndcg(.9,.3)-ndcg(.8,.3)); first within-collection mean, then collection-mean values")
for f in names:
    wc=[]
    for n in DS:
        qs=list(F[n]); x=np.array([F[n][q][f] for q in qs]); y=np.array([Q[n][(.9,.3)][q]-Q[n][(.8,.3)][q] for q in qs])
        wc.append(0 if x.std()==0 or y.std()==0 else np.corrcoef(x,y)[0,1])
    mv=[np.mean([F[n][q][f] for q in F[n]]) for n in DS]; dv=[np.mean([Q[n][(.9,.3)][q]-Q[n][(.8,.3)][q] for q in F[n]]) for n in DS]
    print(f.ljust(8),"within %.3f"%np.mean(wc),"between %.2f"%np.corrcoef(mv,dv)[0,1]," ".join(f"{v:.2f}" for v in mv))
print("delta".ljust(8)," ".join(f"{v:+.3f}" for v in [np.mean([Q[n][(.9,.3)][q]-Q[n][(.8,.3)][q] for q in F[n]]) for n in DS]))
