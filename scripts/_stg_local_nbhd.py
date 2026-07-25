"""STG clean discriminator: does the LOCAL neighbourhood of the valid population (members within the
local-linear regime of a reference) align with the LOCAL Hessian sloppy subspace?
If YES -> prediction holds locally and the global null was a curvature/scale artifact (method problem).
If NO  -> the population's variation does not follow behavioural sloppiness even locally (prediction problem).
Uses the population (no sim) + a local GN Hessian (62 sims). Run in stg env."""
import numpy as np, pandas as pd, json, sys, types
_t=types.ModuleType('torch'); _t.Tensor=type('T',(),{}); _t.as_tensor=_t.tensor=lambda *a,**k:None
sys.modules['torch']=_t
_su=types.ModuleType('sbi.utils'); _su.BoxUniform=type('B',(),{}); sys.modules['sbi']=types.ModuleType('sbi'); sys.modules['sbi.utils']=_su
from scipy import stats
from pyloric import simulate, summary_stats
np.random.seed(0)
X=np.load("/root/e1_valid_params.npy")
NAMES=['AB/PD.Na','AB/PD.CaT','AB/PD.CaS','AB/PD.A','AB/PD.KCa','AB/PD.Kd','AB/PD.H','AB/PD.Leak','LP.Na','LP.CaT','LP.CaS','LP.A','LP.KCa','LP.Kd','LP.H','LP.Leak','PY.Na','PY.CaT','PY.CaS','PY.A','PY.KCa','PY.Kd','PY.H','PY.Leak','Synapses.AB-LP','Synapses.PD-LP','Synapses.AB-PY','Synapses.PD-PY','Synapses.LP-PD','Synapses.LP-PY','Synapses.PY-LP']
COLS=pd.MultiIndex.from_tuples([tuple(n.split('.',1)) for n in NAMES]); D=len(NAMES); TMAX=8000
mu=X.mean(0); sd=X.std(0)+1e-12; Z=(X-mu)/sd
def featsz(z):
    try: return summary_stats(simulate(pd.DataFrame([mu+sd*z],columns=COLS),t_max=TMAX)).to_numpy().ravel()
    except Exception: return np.full(15,np.nan)
ri=int(np.argmin(((X-mu)**2).sum(1))); z_ref=Z[ri]
# local GN Hessian at ref
DELTA=0.015; base=featsz(z_ref); cols=[]
for j in range(D):
    zp=z_ref.copy(); zp[j]+=DELTA; zm=z_ref.copy(); zm[j]-=DELTA
    cols.append((featsz(zp)-featsz(zm))/(2*DELTA))
J=np.array(cols).T; keep=np.isfinite(base)
Jf=np.nan_to_num(J[keep]/(np.nanstd(J[keep],axis=1,keepdims=True)+1e-9),nan=0.0)
H=Jf.T@Jf; w,V=np.linalg.eigh(H); o=np.argsort(w)[::-1]; w=w[o]; V=V[:,o]
k=int(np.searchsorted(np.cumsum(np.abs(w))/np.abs(w).sum(),0.90)+1)
# distances of all population members to ref
dist=np.linalg.norm(Z-z_ref,axis=1); order=np.argsort(dist)
print(f"distance to ref (z): min {dist[order[1]]:.2f}  median {np.median(dist):.2f}  nearest-50 max {dist[order[50]]:.2f}")
res={}
def test_neighbourhood(N):
    nb=order[1:N+1]                       # N nearest (exclude ref itself)
    Dz=Z[nb]-z_ref                         # local displacements
    Sig=np.cov(Dz.T)
    rv=np.array([V[:,j]@Sig@V[:,j] for j in range(D)])
    rho,p=stats.spearmanr(w,rv)            # predict NEGATIVE (local variance in sloppy)
    ratio=rv[k:].mean()/rv[:k].mean()
    # permutation on eigvec assignment
    rng=np.random.RandomState(0); ge=0; NP=20000
    for _ in range(NP):
        pi=rng.permutation(D); Sp=Sig[np.ix_(pi,pi)]
        r=np.array([V[:,j]@Sp@V[:,j] for j in range(D)])
        if r[k:].mean()/r[:k].mean()>=ratio: ge+=1
    return {'N':N,'radius_z':float(dist[order[N]]),'spearman_rho':float(rho),'p':float(p),
            'sloppy_over_stiff_ratio':float(ratio),'perm_p':float((ge+1)/(NP+1))}
for N in [50,100,200,400]:
    r=test_neighbourhood(N); res[f'N{N}']=r
    print(f"N={N} (radius {r['radius_z']:.2f}z): Spearman rho={r['spearman_rho']:+.3f} p={r['p']:.3f} | sloppy/stiff {r['sloppy_over_stiff_ratio']:.2f}x perm_p={r['perm_p']:.3f}")
out={'experiment':'STG_local_neighbourhood_alignment','stiff_dim':k,'predict':'rho<0, ratio>1, perm_p<0.05 if prediction holds locally','results':res}
json.dump(out,open('/root/paper2_STG_local_nbhd.json','w'),indent=2)
print("SAVED")
