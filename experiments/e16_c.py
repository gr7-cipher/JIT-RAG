import sys; sys.path.insert(0,'.')
from e16_lab import *
Q,F=pickle.load(open(f"{R}/lab_b.pkl","rb")); K=list(Q[DS[0]])
M={n:{k:np.mean(list(Q[n][k].values())) for k in K} for n in DS}; Best={n:max(M[n].values()) for n in DS}
print("cfg  macro  worst-regret  mean-regret  fiqa  (relative regret = 1-score/best)")
rows=[]
for k in K:
    reg=[1-M[n][k]/Best[n] for n in DS]; rows.append((max(reg),k,np.mean([M[n][k] for n in DS]),np.mean(reg)))
for w,k,mc,mr in sorted(rows,key=lambda x:x[0])[:10]: print(k,round(mc,4),round(w,4),round(mr,4),round(M["fiqa"][k],4))
print(".8/.3 worst",round(max(1-M[n][(.8,.3)]/Best[n] for n in DS),4))
# leave-one-collection-out on dev
def macro_sel(tr): return max(K,key=lambda k:np.mean([M[n][k] for n in tr]))
def minimax_sel(tr): return min(K,key=lambda k:max(1-M[n][k]/Best[n] for n in tr))
FEATS=["rho","bgap","ov10","midf"]
def adaptive(q_f,f,a0,b,g,mu): return min(1.0,max(.7,a0+b*(q_f-mu)))
def near(a): return min(A_,key=lambda x:abs(x-a))
A_=[.7,.75,.8,.85,.9,.95,1.0]
res={"macro":[],"minimax":[],**{f:[] for f in FEATS}}; base=[]
for te in DS:
    tr=[n for n in DS if n!=te]
    res["macro"].append(M[te][macro_sel(tr)]); res["minimax"].append(M[te][minimax_sel(tr)]); base.append(M[te][(.8,.3)])
    for f in FEATS:
        vals=[F[n][q][f] for n in tr for q in F[n]]; mu=np.mean(vals); sd=np.std(vals)+1e-9; best=None
        for a0 in [.8,.85,.9]:
            for b in [-.1,-.05,-.025,.025,.05,.1]:
                sc=np.mean([np.mean([Q[n][(near(adaptive((F[n][q][f]-mu)/sd,f,a0,b,.3,0)),.3)][q] for q in F[n]]) for n in tr])
                if best is None or sc>best[0]: best=(sc,a0,b)
        _,a0,b=best; res[f].append(np.mean([Q[te][(near(adaptive((F[te][q][f]-mu)/sd,f,a0,b,.3,0)),.3)][q] for q in F[te]]))
print("\nLODO on dev"," ".join(n[:6] for n in DS),"macro")
print("fixed .8/.3".ljust(12)," ".join(f"{x:.3f} " for x in base),round(np.mean(base),4))
for k,v in res.items(): print(k.ljust(12)," ".join(f"{x:.3f} " for x in v),round(np.mean(v),4))
