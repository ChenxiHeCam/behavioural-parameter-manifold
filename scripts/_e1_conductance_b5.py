"""Conductance-space B5 test (answers: was mRNA the wrong data?).
The worm B5 used mRNA expression (a proxy for conductance) and was null. Here we test the SAME
prediction in true CONDUCTANCE space, on the field's gold-standard degeneracy data: the Prinz-Marder
valid pyloric population (2365 conductance sets producing the same rhythm). Two INDEPENDENT measures:
  - behavioural stiffness from the rhythm-feature Hessian (e1b, curvature-based; independent of the population)
  - how much each conductance VARIES across the valid population (CV; independent of the Hessian)
Prediction: behaviourally-sloppy conductances vary MORE across the valid population than stiff ones.
"""
import json, numpy as np
from scipy import stats

X = np.load("e1_valid_params.npy")              # 2365 x 31 valid pyloric conductance sets
e = json.load(open("e1b_results/e1b_hessian.json"))
names = e["param_names"]
cv = X.std(0) / (np.abs(X.mean(0)) + 1e-12)     # per-conductance variability across the valid population

# aggregate Hessian-based stiff/sloppy membership across the 8 rhythm-gated points
def agg(field):
    score = {n: 0.0 for n in names}
    for pt in e[field]:                          # per point: [name, |loading|]
        if isinstance(pt, (list, tuple)) and len(pt) == 2 and isinstance(pt[0], str):
            score[pt[0]] = score.get(pt[0], 0) + abs(pt[1])
    return score
stiff_score = agg("stiff_conductances_top8_by_top_eigvec_loading")     # load on stiffest (high-lambda) eigvecs
sloppy_score = agg("sloppy_conductances_top8_by_bottom_eigvec_loading") # load on sloppiest (low-lambda) eigvecs

# label each conductance by which it loads on more (net behavioural stiffness)
net_stiff = np.array([stiff_score[n] - sloppy_score[n] for n in names])
stiff_idx = np.where(net_stiff > 0)[0]
sloppy_idx = np.where(net_stiff < 0)[0]
cv_stiff = cv[stiff_idx]; cv_sloppy = cv[sloppy_idx]

U, p_mw = stats.mannwhitneyu(cv_sloppy, cv_stiff, alternative="greater")
rho, p_sp = stats.spearmanr(net_stiff, cv)       # predict NEGATIVE: more stiff -> lower CV

# fast voltage-gated Na (the literature's compensated / sloppy current) vs leak (stiff) — named check
na_idx = [i for i, n in enumerate(names) if n.endswith(".Na")]
leak_idx = [i for i, n in enumerate(names) if n.endswith(".Leak")]

out = {
  "experiment": "E1_conductance_space_B5 (Prinz-Marder valid pyloric population, conductance space)",
  "n_valid_models": int(X.shape[0]), "n_conductances": len(names),
  "median_CV_behaviourally_sloppy": float(np.median(cv_sloppy)),
  "median_CV_behaviourally_stiff": float(np.median(cv_stiff)),
  "n_sloppy": int(len(sloppy_idx)), "n_stiff": int(len(stiff_idx)),
  "mannwhitney_sloppy_higher_CV_p": float(p_mw),
  "spearman_netstiffness_vs_CV": {"rho": float(rho), "p": float(p_sp), "predict": "NEGATIVE"},
  "named_check": {
    "median_CV_fastNa": float(np.median(cv[na_idx])),
    "median_CV_leak": float(np.median(cv[leak_idx])),
    "fastNa_over_leak_CV_ratio": float(np.median(cv[na_idx]) / np.median(cv[leak_idx])),
  },
}
json.dump(out, open("NEXT_PAPER_manifold_subspace/paper2_E1_conductance_B5.json", "w"), indent=2)
print(f"valid population: {X.shape[0]} models x {len(names)} conductances")
print(f"median CV  sloppy={np.median(cv_sloppy):.3f} (n={len(sloppy_idx)})  vs  stiff={np.median(cv_stiff):.3f} (n={len(stiff_idx)})")
print(f"Mann-Whitney (sloppy CV > stiff CV) p={p_mw:.4f}")
print(f"Spearman net-stiffness vs CV rho={rho:+.3f} p={p_sp:.4f} (predict NEGATIVE)")
print(f"named: fast-Na CV {np.median(cv[na_idx]):.3f} vs leak CV {np.median(cv[leak_idx]):.3f} -> ratio {np.median(cv[na_idx])/np.median(cv[leak_idx]):.2f}x")
print("SAVED NEXT_PAPER_manifold_subspace/paper2_E1_conductance_B5.json")
