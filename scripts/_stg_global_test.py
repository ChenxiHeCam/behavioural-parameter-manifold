"""STG: (1) validate that local Hessian sloppy/stiff directions are meaningful, (2) the DIRECT global
test of the prediction, by simulation (no local-Hessian-eigenvector basis comparison).
Metric: behavioural distance bdist(theta) = ||(features-ref)/featstd|| (capped if rhythm breaks -> NaN).
All perturbations in z-scored conductance space (population std units) so directions are comparable.
Run in stg env: PYTHONPATH=/root/pyloric_src python _stg_global_test.py"""
import numpy as np, pandas as pd, json, time, sys, types
_t=types.ModuleType('torch'); _t.Tensor=type('T',(),{}); _t.as_tensor=_t.tensor=lambda *a,**k:None
sys.modules['torch']=_t
_su=types.ModuleType('sbi.utils'); _su.BoxUniform=type('B',(),{}); sys.modules['sbi']=types.ModuleType('sbi'); sys.modules['sbi.utils']=_su
from scipy import stats
from pyloric import simulate, summary_stats

X=np.load("/root/e1_valid_params.npy")
NAMES=['AB/PD.Na','AB/PD.CaT','AB/PD.CaS','AB/PD.A','AB/PD.KCa','AB/PD.Kd','AB/PD.H','AB/PD.Leak','LP.Na','LP.CaT','LP.CaS','LP.A','LP.KCa','LP.Kd','LP.H','LP.Leak','PY.Na','PY.CaT','PY.CaS','PY.A','PY.KCa','PY.Kd','PY.H','PY.Leak','Synapses.AB-LP','Synapses.PD-LP','Synapses.AB-PY','Synapses.PD-PY','Synapses.LP-PD','Synapses.LP-PY','Synapses.PY-LP']
COLS=pd.MultiIndex.from_tuples([tuple(n.split('.',1)) for n in NAMES]); D=len(NAMES); TMAX=8000; CAP=12.0
mu=X.mean(0); sd=X.std(0)+1e-12

def feats(theta):
    return summary_stats(simulate(pd.DataFrame([np.asarray(theta,float)],columns=COLS),t_max=TMAX)).to_numpy().ravel()

# reference: population-central valid model
ri=int(np.argmin(((X-mu)**2).sum(1))); z_ref=(X[ri]-mu)/sd
s_ref=feats(X[ri]); fin=np.isfinite(s_ref)
fstd=np.nanstd(np.array([feats(X[j]) for j in range(20)]),axis=0)   # feature scale from 20 valid sets
fstd=np.where(np.isfinite(fstd)&(fstd>0),fstd,1.0)

def bdist(zvec):
    try:
        theta=mu+sd*zvec
        s=feats(theta)
        if not np.all(np.isfinite(s[fin])): return CAP        # rhythm broke -> max behavioural distance
        return float(min(CAP, np.linalg.norm((s[fin]-s_ref[fin])/fstd[fin])/np.sqrt(fin.sum())))
    except Exception:
        return CAP                                            # simulation failed (left valid regime) -> max distance

def move(direction, alpha):           # symmetric behavioural change moving +/- alpha (pop-std units) along unit direction
    u=direction/ (np.linalg.norm(direction)+1e-12)
    return 0.5*(bdist(z_ref+alpha*u)+bdist(z_ref-alpha*u))

# ---- local Hessian in z-space (GN) ----
def feats_z(z):
    try: return feats(mu+sd*z)
    except Exception: return np.full(D if False else len(s_ref), np.nan)
DELTA=0.015
base=feats_z(z_ref); cols=[]
for j in range(D):
    zp=z_ref.copy(); zp[j]+=DELTA; zm=z_ref.copy(); zm[j]-=DELTA
    cols.append((feats_z(zp)-feats_z(zm))/(2*DELTA))
J=np.array(cols).T; keep=np.isfinite(base)
Jf=np.nan_to_num(J[keep]/(np.nanstd(J[keep],axis=1,keepdims=True)+1e-9),nan=0.0)
H=Jf.T@Jf; w,V=np.linalg.eigh(H); o=np.argsort(w)[::-1]; w=w[o]; V=V[:,o]
h_stiff=V[:,0]; h_sloppy=V[:,-1]

t0=time.time()
# ---- (1) VALIDATION + local-vs-global: behavioural change along stiff vs sloppy eigvec at increasing distance ----
alphas=[0.05,0.1,0.2,0.5]
valid={'alpha':alphas,
       'stiff_eigvec_bdist':[round(move(h_stiff,a),3) for a in alphas],
       'sloppy_eigvec_bdist':[round(move(h_sloppy,a),3) for a in alphas]}
print("VALIDATION (behavioural change vs distance):",flush=True)
print("  alpha       ",alphas)
print("  stiff  bdist",valid['stiff_eigvec_bdist'])
print("  sloppy bdist",valid['sloppy_eigvec_bdist'],f"  ({time.time()-t0:.0f}s)",flush=True)

# ---- (2) DIRECT GLOBAL TEST: population-PC directions vs behavioural change ----
Z=(X-mu)/sd; SIG=np.cov(Z.T); pw,PU=np.linalg.eigh(SIG); po=np.argsort(pw)[::-1]; pw=pw[po]; PU=PU[:,po]
ALPHA=0.1
db=np.array([move(PU[:,i],ALPHA) for i in range(D)])     # behavioural change along population PC i
rho,p=stats.spearmanr(pw,db)                              # predict NEGATIVE: high pop-variance -> low behavioural change (sloppy)
k=8
print(f"\nGLOBAL TEST: behavioural change along population PCs vs PC variance",flush=True)
print(f"  top-{k} popvar PCs mean bdist = {db[:k].mean():.3f}  |  bottom-{k} PCs mean bdist = {db[-k:].mean():.3f}")
print(f"  Spearman(pop_variance, behavioural_change) rho={rho:+.3f} p={p:.4f}  (predict NEGATIVE = biology varies where behaviour is flat)")

out={'experiment':'STG_global_direct_test',
 'validation_local_hessian':valid,
 'validation_verdict':'sloppy eigvec should give SMALLER behavioural change than stiff at small alpha if our sloppy direction is correct',
 'global_test':{'alpha_popstd':ALPHA,'spearman_popvar_vs_bdist':{'rho':float(rho),'p':float(p),'predict':'NEGATIVE'},
   'top8_popvar_mean_bdist':float(db[:k].mean()),'bottom8_popvar_mean_bdist':float(db[-k:].mean()),
   'bdist_per_PC':[round(float(x),3) for x in db],'popvar_per_PC':[float(x) for x in pw]},
 'elapsed_s':round(time.time()-t0,1)}
json.dump(out,open('/root/paper2_STG_global_test.json','w'),indent=2)
print("SAVED /root/paper2_STG_global_test.json",flush=True)
