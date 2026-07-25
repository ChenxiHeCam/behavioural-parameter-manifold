"""Conductance-space SUBSPACE B5 (the definitive 'try other data' test).
Prinz-Marder valid pyloric population (2365 conductance sets, same rhythm) = real degeneracy in
CONDUCTANCE space (no mRNA proxy). Compute the rhythm-feature GN Hessian eigenvectors at reference
valid points (INDEPENDENT of the population), then project the population covariance onto them.
Prediction: the population concentrates its variance in the SLOPPY (low-eigenvalue) directions.
Run in stg env: PYTHONPATH=/root/pyloric_src python _stg_subspace_b5.py"""
import numpy as np, pandas as pd, json, time, sys, types
_t = types.ModuleType('torch')                      # stub: pyloric imports torch+sbi only for create_prior (unused here)
_t.Tensor = type('Tensor', (), {}); _t.as_tensor = _t.tensor = lambda *a, **k: None
sys.modules['torch'] = _t
_sbiu = types.ModuleType('sbi.utils'); _sbiu.BoxUniform = type('BoxUniform', (), {})
sys.modules['sbi'] = types.ModuleType('sbi'); sys.modules['sbi.utils'] = _sbiu
from scipy import stats
from pyloric import simulate, summary_stats

X = np.load("/root/e1_valid_params.npy")                # 2365 x 31 valid conductance sets
NAMES = ['AB/PD.Na','AB/PD.CaT','AB/PD.CaS','AB/PD.A','AB/PD.KCa','AB/PD.Kd','AB/PD.H','AB/PD.Leak',
 'LP.Na','LP.CaT','LP.CaS','LP.A','LP.KCa','LP.Kd','LP.H','LP.Leak','PY.Na','PY.CaT','PY.CaS','PY.A',
 'PY.KCa','PY.Kd','PY.H','PY.Leak','Synapses.AB-LP','Synapses.PD-LP','Synapses.AB-PY','Synapses.PD-PY',
 'Synapses.LP-PD','Synapses.LP-PY','Synapses.PY-LP']
COLS = pd.MultiIndex.from_tuples([tuple(n.split('.', 1)) for n in NAMES])
D = len(NAMES); DELTA = 0.015; TMAX = 8000

def feats(theta):
    df = pd.DataFrame([np.asarray(theta, float)], columns=COLS)
    s = summary_stats(simulate(df, t_max=TMAX)).to_numpy().ravel()
    return s

def jac_point(theta0):
    base = feats(theta0)
    cols = []
    for j in range(D):
        tp = theta0.copy(); tp[j] *= np.exp(DELTA)
        tm = theta0.copy(); tm[j] *= np.exp(-DELTA)
        cols.append((feats(tp) - feats(tm)) / (2 * DELTA))
    J = np.array(cols).T                                # n_feat x D
    return base, J

# reference points: population medoid-ish (closest to mean) + a few random valid sets
np.random.seed(0)
mean = X.mean(0)
ref_idx = [int(np.argmin(((X - mean)**2).sum(1)))] + list(np.random.choice(len(X), 4, replace=False))

# population covariance in z-scored conductance space (so trace = D)
Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
SIGMA = np.cov(Z.T)

def analyse(theta0):
    base, J = jac_point(theta0)                          # J: n_feat x D, may contain NaN (perturbation broke rhythm)
    keep = np.isfinite(base)                              # keep features defined at the reference
    Jf = J[keep]
    fstd = np.nanstd(Jf, axis=1, keepdims=True) + 1e-9   # per-feature scale (over the finite param-sensitivities)
    Jn = np.nan_to_num(Jf / fstd, nan=0.0)               # NaN sensitivity (broken rhythm) -> 0 contribution (conservative)
    H = Jn.T @ Jn                                        # GN Hessian = sum_f g_f g_f^T (robust to occasional NaN)
    n_used = int(keep.sum())
    w, V = np.linalg.eigh(H); o = np.argsort(w)[::-1]; w = w[o]; V = V[:, o]
    rv = np.array([V[:, k] @ SIGMA @ V[:, k] for k in range(D)])  # population variance along eigvec k
    rho, p = stats.spearmanr(w, rv)                      # predict NEGATIVE (variance in sloppy)
    keff = int(np.searchsorted(np.cumsum(np.abs(w))/np.abs(w).sum(), 0.90) + 1)
    ratio = rv[keff:].mean() / rv[:keff].mean()          # sloppy/stiff population-variance ratio
    return {'n_feat_used': n_used, 'eff_dim90': keff, 'spearman_rho': float(rho), 'p': float(p),
            'sloppy_over_stiff_ratio': float(ratio), 'eigvals_top5': [float(x) for x in w[:5]],
            'rv': [float(x) for x in rv], 'w': [float(x) for x in w]}

t0 = time.time(); results = []
for i, ri in enumerate(ref_idx):
    r = analyse(X[ri].copy()); results.append(r)
    print(f"ref {i} (idx {ri}): n_feat {r['n_feat_used']} eff90 {r['eff_dim90']} | Spearman rho={r['spearman_rho']:+.3f} p={r['p']:.3f} | sloppy/stiff var {r['sloppy_over_stiff_ratio']:.2f}x | {time.time()-t0:.0f}s", flush=True)

rhos = [r['spearman_rho'] for r in results]; ratios = [r['sloppy_over_stiff_ratio'] for r in results]
out = {'experiment': 'STG_conductance_space_SUBSPACE_B5 (Prinz-Marder valid pyloric population)',
  'n_valid': int(X.shape[0]), 'n_conductances': D, 'delta': DELTA, 'n_reference_points': len(ref_idx),
  'per_reference': results,
  'mean_spearman_stiffness_vs_popvar': float(np.mean(rhos)),
  'mean_sloppy_over_stiff_ratio': float(np.mean(ratios)),
  'prediction': 'NEGATIVE spearman / ratio>1 = population variance concentrates in behaviourally-sloppy directions',
  'elapsed_s': round(time.time() - t0, 1)}
json.dump(out, open('/root/paper2_STG_subspace_B5.json', 'w'), indent=2)
print(f"\nMEAN: Spearman rho={np.mean(rhos):+.3f} | sloppy/stiff var ratio={np.mean(ratios):.2f}x  (predict rho<0, ratio>1)")
print("SAVED /root/paper2_STG_subspace_B5.json")
