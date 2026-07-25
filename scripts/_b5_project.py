"""B5 (subspace-level biological validation): does REAL across-cell-type ion-channel
expression covariance concentrate along the model's SLOPPY eigenvectors (prediction) rather
than its stiff ones? This is the COMBINATION/subspace test that BV1 (per-gene CV) could not do.

Inputs (local):
  REPRODUCIBLE/paper2_real_experiments/cengen/barrett_bulk.tsv  (CeNGEN, 4139 genes x 160 type-replicate samples)
  NEXT_PAPER_manifold_subspace/paper2_B5_baai_hessian_eigvec.json  (20-channel Jacobian + eigvecs from )
Output: NEXT_PAPER_manifold_subspace/paper2_B5_subspace_validation.json
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
TSV  = os.path.join(ROOT, "REPRODUCIBLE/paper2_real_experiments/cengen/barrett_bulk.tsv")
B5   = os.path.join(ROOT, "NEXT_PAPER_manifold_subspace/paper2_B5_baai_hessian_eigvec.json")
OUT  = os.path.join(ROOT, "NEXT_PAPER_manifold_subspace/paper2_B5_subspace_validation.json")

# 16 model ion channels -> unique CeNGEN gene (slo-1/slo-2 and nca collapse)
CH2GENE = {
    'gbshl1':'shl-1','gbshk1':'shk-1','gbkvs1':'kvs-1','gbegl2':'egl-2','gbegl36':'egl-36',
    'gbkqt3':'kqt-3','gbegl19':'egl-19','gbunc2':'unc-2','gbcca1':'cca-1',
    'gbslo1_egl19':'slo-1','gbslo1_unc2':'slo-1','gbslo2_egl19':'slo-2','gbslo2_unc2':'slo-2',
    'gbkcnl':'kcnl-1','gbnca':'nca-2','gbirk':'irk-1',
}
GENE2WB = {"egl-19":"WBGene00001187","unc-2":"WBGene00006742","cca-1":"WBGene00000367",
    "egl-2":"WBGene00001171","egl-36":"WBGene00001202","kqt-3":"WBGene00002235",
    "irk-1":"WBGene00002149","nca-2":"WBGene00003558","shk-1":"WBGene00014261",
    "shl-1":"WBGene00022240","kvs-1":"WBGene00002242","kcnl-1":"WBGene00007176",
    "slo-1":"WBGene00004830","slo-2":"WBGene00004831"}
SLOPPY_GENES = {'egl-19','unc-2','cca-1','slo-1','slo-2'}   # fast Ca + Ca-activated K (the sloppy class)

# ---- real: load CeNGEN matrix first (to know which genes are measured) ----
df = pd.read_csv(TSV, sep='\t', index_col=0, engine='python')
df.columns = [c.strip().strip('"') for c in df.columns]
df.index = [str(i).strip().strip('"') for i in df.index]
df = df.astype(float); df = df / df.sum(axis=0) * 1e6          # CPM

# ---- model: aggregate 16-ion Jacobian to unique genes, RESTRICTED to genes present in CeNGEN ----
b5 = json.load(open(B5))
chans = b5['channels']; J = np.array(b5['jacobian'])          # 6 obs x 20 channels
genes = []
for c in CH2GENE:
    g = CH2GENE[c]
    if g not in genes and GENE2WB.get(g) in df.index:         # unique AND measured
        genes.append(g)
dropped = sorted({CH2GENE[c] for c in CH2GENE} - set(genes))
Jg = np.zeros((J.shape[0], len(genes)))
for c in CH2GENE:                                             # sum channels sharing a gene
    g = CH2GENE[c]
    if g in genes:
        Jg[:, genes.index(g)] += J[:, chans.index(c)]
Hg = Jg.T @ Jg
w, V = np.linalg.eigh(Hg); o = np.argsort(w)[::-1]; w = w[o]; V = V[:, o]   # descending
print(f"genes dropped (not in CeNGEN): {dropped}")

# ---- real: WITHIN-TYPE residual covariance (isolate biological freedom-to-vary, remove cell identity) ----
import re
X = np.array([df.loc[GENE2WB[g]].values for g in genes])     # genes x samples
expr = np.log1p(X)                                            # log CPM
cols = list(df.columns)
col_type = [re.sub(r'r\d+$', '', c) for c in cols]
# residual of each replicate from its neuron-type mean (per gene)
resid = np.zeros_like(expr)
for t in set(col_type):
    idx = [i for i, ct in enumerate(col_type) if ct == t]
    if len(idx) >= 2:
        resid[:, idx] = expr[:, idx] - expr[:, idx].mean(axis=1, keepdims=True)
    else:
        resid[:, idx] = np.nan
keep = ~np.isnan(resid).any(axis=0)
resid = resid[:, keep]                                        # genes x (within-type-centred samples)
R = np.corrcoef(resid)                                        # within-type co-variation (the freedom-to-vary signal)
C = np.cov(resid)
# secondary: pooled across-cell-type correlation (dominated by cell identity, reported for contrast)
R_pooled = np.corrcoef(expr)

# ---- projection: real variance along each model eigenvector vs model stiffness ----
real_var_corr = np.array([V[:, j] @ R @ V[:, j] for j in range(len(genes))])
real_var_cov  = np.array([V[:, j] @ C @ V[:, j] for j in range(len(genes))])
stiff_rank = w                                               # eigenvalue = model stiffness of direction j

rho_corr, p_corr = stats.spearmanr(stiff_rank, real_var_corr)   # predict NEGATIVE
rho_cov,  p_cov  = stats.spearmanr(stiff_rank, real_var_cov)
# stiff subspace (model-identified, top-k by 99% mass) vs sloppy/null subspace: mean real variance per direction
k = int(b5['eff_dim_99'])                                       # number of model-stiff directions
stiff_var_per_dir  = float(real_var_corr[:k].mean())
sloppy_var_per_dir = float(real_var_corr[k:].mean())
sloppy_over_stiff_ratio = sloppy_var_per_dir / stiff_var_per_dir
# half/half share (robustness)
half = len(genes)//2
sloppy_share_corr = float(real_var_corr[half:].sum()/real_var_corr.sum())
stiff_share_corr  = float(real_var_corr[:half].sum()/real_var_corr.sum())
# pooled across-cell-type contrast (identity-dominated)
real_var_pooled = np.array([V[:, j] @ R_pooled @ V[:, j] for j in range(len(genes))])
rho_pooled, p_pooled = stats.spearmanr(stiff_rank, real_var_pooled)
# permutation test: is the sloppy/stiff variance ratio larger than if genes were randomly arranged?
rng = np.random.RandomState(0); NP = 20000; ge = 0
for _ in range(NP):
    pidx = rng.permutation(len(genes))
    Rp = R[np.ix_(pidx, pidx)]
    rv = np.array([V[:, j] @ Rp @ V[:, j] for j in range(len(genes))])
    if rv[k:].mean() / rv[:k].mean() >= sloppy_over_stiff_ratio: ge += 1
p_perm = (ge + 1) / (NP + 1)
# per-gene cross-check (BV1-style, expected null): per-gene model stiffness vs per-gene real total corr-variance
per_gene_modelstiff = np.array([(Jg[:, i]**2).sum() for i in range(len(genes))])
per_gene_realvar = np.diag(R)                                # =1 each (correlation) -> use covariance diag
per_gene_realvar_cov = np.diag(C)
rho_pergene, p_pergene = stats.spearmanr(per_gene_modelstiff, per_gene_realvar_cov)

out = {
  'experiment': 'B5_subspace_level_biological_validation',
  'question': 'does real CeNGEN expression covariance concentrate along the model SLOPPY eigenvectors (subspace test, not per-gene)',
  'genes': genes, 'n_genes': len(genes),
  'model_eigenvalues': [float(x) for x in w],
  'real_variance_along_eigvec_corr': [float(x) for x in real_var_corr],
  'real_variance_along_eigvec_cov':  [float(x) for x in real_var_cov],
  'SUBSPACE_spearman_stiffness_vs_realvar_corr': {'rho': float(rho_corr), 'p': float(p_corr), 'predict':'NEGATIVE'},
  'SUBSPACE_spearman_stiffness_vs_realvar_cov':  {'rho': float(rho_cov),  'p': float(p_cov)},
  'stiff_subspace_dim_k': k,
  'stiff_subspace_real_var_per_dir': stiff_var_per_dir,
  'sloppy_subspace_real_var_per_dir': sloppy_var_per_dir,
  'sloppy_over_stiff_var_ratio': sloppy_over_stiff_ratio,
  'sloppy_over_stiff_permutation_p': p_perm,
  'sloppy_half_share_of_real_variance_corr': sloppy_share_corr,
  'stiff_half_share_of_real_variance_corr': stiff_share_corr,
  'pooled_acrosstype_spearman': {'rho': float(rho_pooled), 'p': float(p_pooled),
       'note':'across-cell-type (identity-dominated) contrast'},
  'per_gene_control_spearman_cov': {'rho': float(rho_pergene), 'p': float(p_pergene),
       'note':'BV1-style per-gene test, expected weak/null by construction'},
  'genes_dropped_not_in_cengen': dropped,
  'sloppy_class_genes': sorted(SLOPPY_GENES),
  'VERDICT': ('NULL / inconclusive with available data: the within-type CeNGEN expression covariance does NOT '
              'concentrate in the model sloppy subspace beyond chance (permutation p=%.2f; sloppy/stiff var ratio '
              '%.2fx is not above random gene arrangements). The directional Spearman is weakly negative (rho=%.2f) '
              'and the per-gene control is null as predicted, but the subspace prediction is NOT positively confirmed. '
              'Caveat: CeNGEN measures cross-cell-type replicate variability, a proxy that is dominated by technical/'
              'cell-identity variation rather than the cross-individual conductance freedom the prediction concerns, '
              'and 4/14 channel genes (incl. the sloppy-class unc-2) are absent. A cross-individual conductance-level '
              'dataset remains the genuinely outstanding test.') % (p_perm, sloppy_over_stiff_ratio, rho_corr),
  'caveats': ['CeNGEN expression is across cell-type replicates (proxy for biological variability), '
              'not across whole individuals; 4 of 14 channel genes (unc-2, shk-1, shl-1, kcnl-1) absent '
              'from the CeNGEN tables; model Hessian from 6 coarse behavioural observables; '
              'gene->channel aggregation hand-curated; n=10 genes limits power.'],
}
json.dump(out, open(OUT, 'w'), indent=2)
print(f"genes ({len(genes)}): {genes}")
print(f"model eigenvalues (desc): {[round(float(x),2) for x in w]}")
print(f"real var along eigvec (corr): {[round(float(x),2) for x in real_var_corr]}")
print(f"\nSUBSPACE TEST (within-type residual covariance):")
print(f"  stiffness vs real-variance Spearman rho={rho_corr:+.3f} p={p_corr:.4f} (predict NEGATIVE)")
print(f"  sloppy subspace var/dir = {sloppy_var_per_dir:.3f}  vs stiff subspace var/dir = {stiff_var_per_dir:.3f}  -> ratio {sloppy_over_stiff_ratio:.2f}x  (permutation p={p_perm:.4f})")
print(f"  sloppy-half share of real variance = {sloppy_share_corr:.2f}  (stiff-half = {stiff_share_corr:.2f})")
print(f"  pooled across-type (identity-dominated) rho={rho_pooled:+.3f} p={p_pooled:.3f}")
print(f"per-gene control (expect null): rho={rho_pergene:+.3f} p={p_pergene:.3f}")
print("SAVED", OUT)
