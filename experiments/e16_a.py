import sys, time; sys.path.insert(0,'.')
from e16_lab import *
ALL={n:load(n) for n in DS}; print("loaded",flush=True)
cfg=[("af",.8,.3),("af",.8,0),("af",.9,.3),("af",.9,0),("af",.95,.3),("af",.95,0),("af",1,0),("h50",.9,0),("h100",.9,0),("h50",.8,.3),("h100",.8,.3),("af100",.8,.3),("af100",.9,.3),("h50",.95,0),("h100",.95,0),("h100",1,0)]
print("cfg".ljust(16),"  ".join(n[:6].ljust(11) for n in DS),"macro",flush=True)
for c in cfg:
    t=time.time(); v=[ev(run(ALL[n],*c),ALL[n]["qrels"])[:2] for n in DS]
    print(f"{c[0]}|{c[1]}|{c[2]}".ljust(16),"  ".join(f"{a:.3f}/{b:.3f}" for a,b in v),f"{np.mean([a for a,b in v]):.4f} {np.mean([b for a,b in v]):.4f}",round(time.time()-t,1),flush=True)
