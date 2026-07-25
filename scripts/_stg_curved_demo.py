"""STG: does the local-linear sloppy method MISS a real global behaviour-preserving structure?
Direct, non-circular demonstration:
 (A) real degeneracy population members: large PARAMETER move from ref, but tiny BEHAVIOURAL change.
 (B) random conductance directions of the SAME parameter magnitude: large behavioural change (break rhythm).
 (C) the local-Hessian SLOPPY subspace: does the population actually lie in it? (the B5 local test said ~no)
If A<<B but the population is NOT in the local sloppy subspace -> the global behaviour-preserving structure
is real (prediction's core holds) but invisible to local-linear sloppy (= the B5 null was a method artifact).
Run in stg env: PYTHONPATH=/root/pyloric_src python _stg_curved_demo.py"""
import numpy as np, pandas as pd, json, time, sys, types
_t=types.ModuleType('torch'); _t.Tensor=type('T',(),{}); _t.as_tensor=_t.tensor=lambda *a,**k:None
sys.modules['torch']=_t
_su=types.ModuleType('sbi.utils'); _su.BoxUniform=type('B',(),{}); sys.modules['sbi']=types.ModuleType('sbi'); sys.modules['sbi.utils']=_su
from scipy import stats
from pyloric import simulate, summary_stats
np.random.seed(0)

X=np.load("/root/e1_valid_params.npy")
NAMES=['AB/PD.Na','AB/PD.CaT','AB/PD.CaS','AB/PD.A','AB/PD.KCa','AB/PD.Kd','AB/PD.H','AB/PD.Leak','LP.Na','LP.CaT','LP.CaS','LP.A','LP.KCa','LP.Kd','LP.H','LP.Leak','PY.Na','PY.CaT','PY.CaS','PY.A','PY.KCa','PY.Kd','PY.H','PY.Leak','Synapses.AB-LP','Synapses.PD-LP','Synapses.AB-PY','Synapses.PD-PY','Synapses.LP-PD','Synapses.LP-PY','Synapses.PY-LP']
COLS=pd.MultiIndex.from_tuples([tuple(n.split('.',1)) for n in NAMES]); D=len(NAMES); TMAX=8000; CAP=12.0
mu=X.mean(0); sd=X.std(0)+1e-12; Z=(X-mu)/sd

def feats(theta):
    try: return summary_stats(simulate(pd.DataFrame([np.asarray(theta,float)],columns=COLS),t_max=TMAX)).to_numpy().ravel()
    except Exception: return np.full(15,np.nan)
ri=int(np.argmin(((X-mu)**2).sum(1))); s_ref=feats(X[ri]); fin=np.isfinite(s_ref)
fstd=np.nanstd(np.array([feats(X[j]) for j in range(20)]),axis=0); fstd=np.where(np.isfinite(fstd)&(fstd>0),fstd,1.0)
def bdist_theta(theta):
    s=feats(theta)
    if not np.all(np.isfinite(s[fin])): return CAP
    return float(min(CAP, np.linalg.norm((s[fin]-s_ref[fin])/fstd[fin])/np.sqrt(fin.sum())))

z_ref=Z[ri]; t0=time.time()
# sample population members + their z-distance from ref
idx=np.random.choice(len(X),80,replace=False)
pop_pdist=[]; pop_bdist=[]; rnd_bdist=[]
for k in idx:
    zk=Z[k]; pd_=float(np.linalg.norm(zk-z_ref))
    pop_pdist.append(pd_); pop_bdist.append(bdist_theta(X[k]))
    # random direction of the SAME z-distance from ref
    r=np.random.randn(D); r=r/np.linalg.norm(r)*pd_
    rnd_bdist.append(bdist_theta(mu+sd*(z_ref+r)))
pop_pdist=np.array(pop_pdist); pop_bdist=np.array(pop_bdist); rnd_bdist=np.array(rnd_bdist)

# local Hessian sloppy/stiff subspace at ref (z-space) -- do population displacements lie in sloppy half?
DELTA=0.015; base=feats(mu+sd*z_ref); cols=[]
for j in range(D):
    zp=z_ref.copy(); zp[j]+=DELTA; zm=z_ref.copy(); zm[j]-=DELTA
    cols.append((feats(mu+sd*zp)-feats(mu+sd*zm))/(2*DELTA))
J=np.array(cols).T; keep=np.isfinite(base)
Jf=np.nan_to_num(J[keep]/(np.nanstd(J[keep],axis=1,keepdims=True)+1e-9),nan=0.0)
H=Jf.T@Jf; w,V=np.linalg.eigh(H); o=np.argsort(w)[::-1]; w=w[o]; V=V[:,o]
ks=int(np.searchsorted(np.cumsum(np.abs(w))/np.abs(w).sum(),0.90)+1)   # stiff subspace dim
# fraction of each population displacement's norm lying in the STIFF subspace (should be SMALL if pop is sloppy)
Vst=V[:,:ks]
frac_in_stiff=[]
for k in idx:
    d=Z[k]-z_ref; d=d/(np.linalg.norm(d)+1e-12)
    frac_in_stiff.append(float(np.linalg.norm(Vst.T@d)**2))   # 0=fully in sloppy subspace, 1=fully stiff
frac_in_stiff=np.array(frac_in_stiff)
expected_random_frac=ks/D    # if displacements were random, this fraction would lie in the ks-dim stiff subspace

out={'experiment':'STG_curved_vs_local_demo',
 'n_sampled':len(idx),
 'population_param_dist_median': float(np.median(pop_pdist)),
 'population_behav_dist_median': float(np.median(pop_bdist)),
 'random_matched_behav_dist_median': float(np.median(rnd_bdist)),
 'random_over_population_behav_ratio': float(np.median(rnd_bdist)/max(np.median(pop_bdist),1e-3)),
 'frac_pop_displacement_in_local_STIFF_subspace_median': float(np.median(frac_in_stiff)),
 'frac_expected_if_random': float(expected_random_frac),
 'local_stiff_subspace_dim': ks,
 'elapsed_s': round(time.time()-t0,1)}
print(f"population: param-dist median {np.median(pop_pdist):.2f} (z-units), behav-dist median {np.median(pop_bdist):.3f}")
print(f"random matched-magnitude: behav-dist median {np.median(rnd_bdist):.3f}  -> random/pop ratio {out['random_over_population_behav_ratio']:.1f}x")
print(f"frac of population displacement in LOCAL STIFF subspace: median {np.median(frac_in_stiff):.3f}  (random expectation {expected_random_frac:.3f}, stiff dim {ks}/{D})")
json.dump(out,open('/root/paper2_STG_curved_demo.json','w'),indent=2)
print("SAVED /root/paper2_STG_curved_demo.json")
