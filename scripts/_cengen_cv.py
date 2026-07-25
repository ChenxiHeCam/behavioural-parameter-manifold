"""Paper2 BV1: CeNGEN ion-channel expression variability vs model stiff/sloppy.
The CORRECT (graded-variation) replacement for the failed G1 null-mutant test. The framework
predicts robustness to GRADED variation (Marder homeostatic compensation), so a sloppy channel
(behaviour-invariant to graded conductance change) should tolerate MORE expression variability.
Test: per-channel-gene within-neuron-type expression CV (CeNGEN scRNAseq, 128 neuron types,
per-replicate columns) vs the model per-channel behavioural stiffness. Prediction: stiffness UP
-> CV DOWN (stiff channels tightly regulated; sloppy channels free to vary).
Controls for expression level (CV ~ mean is the classic confound).
"""
import urllib.request, json, urllib.parse, re, sys
import numpy as np, pandas as pd
from scipy import stats

TMM = '/root/autodl-tmp/cengen/barrett_bulk.tsv'  # raw counts, 4139 genes x 160 per-type replicate samples

# model per-channel behavioural stiffness (paper2_baai_perchannel_hessian.json, per_channel_stiffness)
MODEL_STIFF = {  # gene -> stiffness (high=stiff, low=sloppy)
    'egl-19': 0.492, 'unc-2': 0.311, 'cca-1': 0.857, 'egl-2': 1.530, 'egl-36': 2.877,
    'kqt-3': 1.225, 'irk-1': 1.537, 'nca-1': 1.659, 'nca-2': 1.659, 'shk-1': 0.440,
    'shl-1': 0.560, 'kvs-1': 0.512, 'kcnl-1': 0.998,
    'slo-1': 0.119,   # avg(gbslo1_egl19 0.009, gbslo1_unc2 0.229)
    'slo-2': 0.209,   # avg(gbslo2_egl19 0.326, gbslo2_unc2 0.093)
    'unc-9': 2.539, 'unc-7': 2.539,   # gap junction (stiff)
}
# sloppy class = fast Ca + Ca-activated K (the sloppy axes); stiff class = the rest
SLOPPY = {'egl-19', 'unc-2', 'cca-1', 'slo-1', 'slo-2'}


def mg(name):
    url = 'https://mygene.info/v3/query?' + urllib.parse.urlencode(
        {'q': 'symbol:' + name, 'species': '6239', 'fields': 'symbol,WormBase'})
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'g'}), timeout=30))
        for h in d.get('hits', []):
            wb = h.get('WormBase')
            if wb:
                return wb if isinstance(wb, str) else wb[0]
    except Exception as e:
        print('mg err', name, repr(e)[:40], flush=True)
    return None


gene2wb = {g: mg(g) for g in MODEL_STIFF}
print('resolved:', {g: gene2wb[g] for g in gene2wb}, flush=True)

print('loading CeNGEN TMM matrix ...', flush=True)
df = pd.read_csv(TMM, sep='\t', index_col=0, engine='python')
df.columns = [c.strip().strip('"') for c in df.columns]
df.index = [str(i).strip().strip('"') for i in df.index]
# raw counts -> CPM (normalize sequencing depth per sample before computing CV)
df = df.astype(float)
df = df / df.sum(axis=0) * 1e6
print('matrix:', df.shape, '(genes x neuron-replicate samples), CPM-normalised', flush=True)

# group columns by neuron type (strip trailing replicate marker rNNN)
col_type = {c: re.sub(r'r\d+$', '', c) for c in df.columns}
types = {}
for c, t in col_type.items():
    types.setdefault(t, []).append(c)
types = {t: cs for t, cs in types.items() if len(cs) >= 2}
print('neuron types with >=2 reps:', len(types), flush=True)


def within_type_cv(wb):
    if wb is None or wb not in df.index:
        return None, None, 0
    row = df.loc[wb]
    cvs, means = [], []
    for t, cs in types.items():
        v = row[cs].astype(float).values
        m = np.nanmean(v)
        if m > 5:  # only types where the gene is expressed (avoid noise-dominated CV)
            cvs.append(np.nanstd(v) / m); means.append(m)
    if len(cvs) < 3:
        return None, None, len(cvs)
    return float(np.median(cvs)), float(np.median(means)), len(cvs)


rows = []
for g, wb in gene2wb.items():
    cv, meanexpr, ntype = within_type_cv(wb)
    rows.append({'gene': g, 'wb': wb, 'stiffness': MODEL_STIFF[g], 'cv': cv,
                 'mean_expr': meanexpr, 'n_types': ntype, 'sloppy': g in SLOPPY})
R = pd.DataFrame(rows)
Rok = R.dropna(subset=['cv']).copy()
print('\n=== per-gene CV ===', flush=True)
print(Rok[['gene', 'stiffness', 'cv', 'mean_expr', 'n_types', 'sloppy']].sort_values('stiffness').to_string(index=False), flush=True)

# primary: stiffness vs CV (predict negative)
rho, p = stats.spearmanr(Rok['stiffness'], Rok['cv'])
# control for expression level: partial Spearman (regress out log mean_expr ranks)
def resid_rank(a, b):
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    bb = np.polyfit(rb, ra, 1); return ra - np.polyval(bb, rb)
res_stiff = resid_rank(Rok['stiffness'], np.log(Rok['mean_expr']))
res_cv = resid_rank(Rok['cv'], np.log(Rok['mean_expr']))
rho_partial, p_partial = stats.spearmanr(res_stiff, res_cv)
# class contrast: sloppy genes higher CV?
cv_sloppy = Rok[Rok['sloppy']]['cv'].values
cv_stiff = Rok[~Rok['sloppy']]['cv'].values
if len(cv_sloppy) and len(cv_stiff):
    U, p_mw = stats.mannwhitneyu(cv_sloppy, cv_stiff, alternative='greater')
else:
    U, p_mw = None, None

out = {
    'experiment': 'BV1_CeNGEN_expression_variability_vs_stiffness',
    'source': 'CeNGEN Average_integrated_TMM (per-neuron-type replicates), gene IDs via mygene.info',
    'prediction': 'stiffness UP -> within-type expression CV DOWN (sloppy channels tolerate more variability)',
    'n_genes': len(Rok),
    'gene2wb': gene2wb,
    'per_gene': Rok[['gene', 'wb', 'stiffness', 'cv', 'mean_expr', 'n_types', 'sloppy']].to_dict('records'),
    'spearman_stiffness_vs_cv': {'rho': float(rho), 'p': float(p), 'note': 'predict negative'},
    'partial_spearman_controlling_expr': {'rho': float(rho_partial), 'p': float(p_partial)},
    'class_contrast_sloppy_higher_cv': {
        'median_cv_sloppy': float(np.median(cv_sloppy)) if len(cv_sloppy) else None,
        'median_cv_stiff': float(np.median(cv_stiff)) if len(cv_stiff) else None,
        'mannwhitney_p_sloppy_greater': float(p_mw) if p_mw is not None else None,
        'n_sloppy': int(len(cv_sloppy)), 'n_stiff': int(len(cv_stiff)),
    },
    'caveats': ['within-type CV is a proxy for cell-to-cell regulatory variability; '
                'CV-mean confound controlled by partial correlation on log mean expression; '
                'gene->channel map hand-curated; n genes small.'],
}
json.dump(out, open('/root/autodl-tmp/paper2_BV1_cengen_cv.json', 'w'), indent=2)
print('\n=== RESULT ===', flush=True)
print('stiffness vs CV Spearman rho=%.3f p=%.3f (predict NEGATIVE)' % (rho, p), flush=True)
print('partial (control expr) rho=%.3f p=%.3f' % (rho_partial, p_partial), flush=True)
print('class contrast: median CV sloppy=%.3f vs stiff=%.3f MW p(sloppy>stiff)=%s' % (
    np.median(cv_sloppy) if len(cv_sloppy) else float('nan'),
    np.median(cv_stiff) if len(cv_stiff) else float('nan'), p_mw), flush=True)
print('BV1_DONE', flush=True)
