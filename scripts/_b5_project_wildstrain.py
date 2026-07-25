"""B5 (wild-isolate, the CORRECT data type): does REAL cross-strain (natural-variation) channel-gene
expression covariance concentrate along the model SLOPPY eigenvectors? Uses CaeNDR / Bell&Paaby 2024
vst expression across 208 wild C. elegans strains (cross-INDIVIDUAL, unlike CeNGEN cross-cell-type).

Inputs (local):
  REPRODUCIBLE/paper2_real_experiments/cengen/cendr_14genes_by_strain.csv  (strains x WBGene, vst meanexp)
  NEXT_PAPER_manifold_subspace/paper2_B5_baai_hessian_eigvec.json
Output: NEXT_PAPER_manifold_subspace/paper2_B5_wildstrain_validation.json
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(ROOT, "REPRODUCIBLE/paper2_real_experiments/cengen/cendr_14genes_by_strain.csv")
B5  = os.path.join(ROOT, "NEXT_PAPER_manifold_subspace/paper2_B5_baai_hessian_eigvec.json")
OUT = os.path.join(ROOT, "NEXT_PAPER_manifold_subspace/paper2_B5_wildstrain_validation.json")

CH2GENE = {'gbshl1':'shl-1','gbshk1':'shk-1','gbkvs1':'kvs-1','gbegl2':'egl-2','gbegl36':'egl-36',
    'gbkqt3':'kqt-3','gbegl19':'egl-19','gbunc2':'unc-2','gbcca1':'cca-1','gbslo1_egl19':'slo-1',
    'gbslo1_unc2':'slo-1','gbslo2_egl19':'slo-2','gbslo2_unc2':'slo-2','gbkcnl':'kcnl-1',
    'gbnca':'nca-2','gbirk':'irk-1'}
GENE2WB = {"egl-19":"WBGene00001187","unc-2":"WBGene00006742","cca-1":"WBGene00000367",
    "egl-2":"WBGene00001171","egl-36":"WBGene00001202","kqt-3":"WBGene00002235","irk-1":"WBGene00002149",
    "nca-2":"WBGene00003558","shk-1":"WBGene00014261","shl-1":"WBGene00022240","kvs-1":"WBGene00002242",
    "kcnl-1":"WBGene00007176","slo-1":"WBGene00004830","slo-2":"WBGene00004831"}
SLOPPY_GENES = {'egl-19','unc-2','cca-1','slo-1','slo-2'}

# real cross-strain expression matrix (strains x WBGene)
M = pd.read_csv(CSV, index_col=0)
present = {g for g, wb in GENE2WB.items() if wb in M.columns}

# model: aggregate 16-ion Jacobian to unique genes present in the wild-strain data
b5 = json.load(open(B5)); chans = b5['channels']; J = np.array(b5['jacobian'])
genes = []
for c in CH2GENE:
    g = CH2GENE[c]
    if g not in genes and g in present: genes.append(g)
dropped = sorted({CH2GENE[c] for c in CH2GENE} - set(genes))
Jg = np.zeros((J.shape[0], len(genes)))
for c in CH2GENE:
    if CH2GENE[c] in genes: Jg[:, genes.index(CH2GENE[c])] += J[:, chans.index(c)]
Hg = Jg.T @ Jg
w, V = np.linalg.eigh(Hg); o = np.argsort(w)[::-1]; w = w[o]; V = V[:, o]
k = int(b5['eff_dim_99'])

# real cross-strain correlation (vst meanexp is already log-scale; drop strains with any NaN)
X = M[[GENE2WB[g] for g in genes]].dropna()
R = np.corrcoef(X.values.T)   # gene x gene, across strains
real_var = np.array([V[:, j] @ R @ V[:, j] for j in range(len(genes))])
rho, p = stats.spearmanr(w, real_var)
stiff_pd = float(real_var[:k].mean()); sloppy_pd = float(real_var[k:].mean())
ratio = sloppy_pd / stiff_pd
# permutation test on the sloppy/stiff ratio
rng = np.random.RandomState(0); NP = 20000; ge = 0
for _ in range(NP):
    pidx = rng.permutation(len(genes)); Rp = R[np.ix_(pidx, pidx)]
    rv = np.array([V[:, j] @ Rp @ V[:, j] for j in range(len(genes))])
    if rv[k:].mean() / rv[:k].mean() >= ratio: ge += 1
p_perm = (ge + 1) / (NP + 1)

out = {'experiment': 'B5_wildstrain_subspace_validation (CaeNDR 208 wild strains, cross-individual)',
  'n_strains': int(X.shape[0]), 'genes': genes, 'genes_dropped': dropped,
  'model_eigenvalues': [float(x) for x in w], 'stiff_subspace_dim_k': k,
  'real_variance_along_eigvec': [float(x) for x in real_var],
  'SUBSPACE_spearman_stiffness_vs_realvar': {'rho': float(rho), 'p': float(p), 'predict': 'NEGATIVE'},
  'stiff_subspace_real_var_per_dir': stiff_pd, 'sloppy_subspace_real_var_per_dir': sloppy_pd,
  'sloppy_over_stiff_var_ratio': ratio, 'sloppy_over_stiff_permutation_p': p_perm,
  'sloppy_class_genes': sorted(SLOPPY_GENES),
  'caveat': 'cross-strain vst meanexp = natural inter-individual expression variation (the correct data '
            'type); gene->channel aggregation hand-curated; model Hessian from 6 coarse behavioural observables.'}
json.dump(out, open(OUT, 'w'), indent=2)
print(f"strains={X.shape[0]}  genes({len(genes)})={genes}  dropped={dropped}")
print(f"model eigenvalues: {[round(float(x),2) for x in w]}")
print(f"real var along eigvec: {[round(float(x),2) for x in real_var]}")
print(f"\nSUBSPACE TEST  stiffness vs real-variance  Spearman rho={rho:+.3f} p={p:.4f} (predict NEGATIVE)")
print(f"  sloppy var/dir {sloppy_pd:.3f} vs stiff {stiff_pd:.3f} -> ratio {ratio:.2f}x  permutation p={p_perm:.4f}")
print("SAVED", OUT)
