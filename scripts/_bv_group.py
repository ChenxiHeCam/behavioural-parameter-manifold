"""Paper2 BV2: GROUP-level biological validation (not per-channel).
The per-channel tests (G1, BV1) failed: behaviour does not carry which-specific-channel info.
But the model's ROBUST, cross-model finding (G2) is at the CLASS level: WIRING (gap junction +
synaptic) is stiff, SINGLE-CELL ION CHANNELS are sloppy. So test biology at the GROUP level:
do ion-channel genes (sloppy class) show MORE expression variability than wiring genes (stiff
class), AS GROUPS -- robust to any individual gene. CeNGEN within-type CV, Barrett bulk.
"""
import urllib.request, json, urllib.parse, re
import numpy as np, pandas as pd
from scipy import stats

TMM = '/root/autodl-tmp/cengen/barrett_bulk.tsv'

# WIRING / structural-connectivity group  ==  STIFF class in the model (gj + syn)
WIRING = ['unc-7', 'unc-9', 'eat-5', 'che-7',
          'inx-1', 'inx-2', 'inx-3', 'inx-4', 'inx-5', 'inx-6', 'inx-7', 'inx-8', 'inx-9',
          'inx-10', 'inx-11', 'inx-12', 'inx-13', 'inx-14', 'inx-16', 'inx-18', 'inx-19', 'inx-22',
          'unc-13', 'unc-18', 'unc-10', 'unc-64', 'rab-3', 'snt-1', 'snb-1', 'ric-4']
# SINGLE-CELL VOLTAGE-GATED ION CHANNEL group  ==  SLOPPY class in the model
CHANNEL = ['egl-19', 'unc-2', 'cca-1',                       # Ca
           'slo-1', 'slo-2', 'kcnl-1', 'kcnl-2', 'kcnl-3',   # Ca-activated K
           'shk-1', 'shl-1', 'kvs-1', 'kvs-4', 'egl-2', 'egl-36', 'exp-2',  # Kv
           'kqt-1', 'kqt-2', 'kqt-3', 'irk-1', 'irk-2', 'irk-3',            # KCNQ / Kir
           'nca-1', 'nca-2', 'unc-77']                       # Na-leak


def mg(name):
    url = 'https://mygene.info/v3/query?' + urllib.parse.urlencode(
        {'q': 'symbol:' + name, 'species': '6239', 'fields': 'WormBase'})
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'g'}), timeout=30))
        for h in d.get('hits', []):
            wb = h.get('WormBase')
            if wb:
                return wb if isinstance(wb, str) else wb[0]
    except Exception:
        pass
    return None


print('resolving genes via mygene.info ...', flush=True)
g2wb = {g: mg(g) for g in WIRING + CHANNEL}

df = pd.read_csv(TMM, sep='\t', index_col=0, engine='python')
df.columns = [c.strip().strip('"') for c in df.columns]
df.index = [str(i).strip().strip('"') for i in df.index]
df = df.astype(float); df = df / df.sum(axis=0) * 1e6
col_type = {c: re.sub(r'r\d+$', '', c) for c in df.columns}
types = {}
for c, t in col_type.items():
    types.setdefault(t, []).append(c)
types = {t: cs for t, cs in types.items() if len(cs) >= 2}


def within_type_cv(wb):
    if wb is None or wb not in df.index:
        return None
    row = df.loc[wb]; cvs = []
    for t, cs in types.items():
        v = row[cs].astype(float).values; m = np.nanmean(v)
        if m > 5:
            cvs.append(np.nanstd(v) / m)
    return float(np.median(cvs)) if len(cvs) >= 3 else None


def group_cv(genes):
    out = {}
    for g in genes:
        cv = within_type_cv(g2wb.get(g))
        if cv is not None:
            out[g] = cv
    return out


wiring_cv = group_cv(WIRING)
channel_cv = group_cv(CHANNEL)
wv = list(wiring_cv.values()); cv = list(channel_cv.values())
U, p = stats.mannwhitneyu(cv, wv, alternative='greater')  # predict channel CV > wiring CV

out = {
    'experiment': 'BV2_group_level_expression_variability',
    'prediction': 'single-cell ion-channel genes (sloppy class) have HIGHER within-type expression CV than wiring genes (gap-junction + synaptic, stiff class)',
    'source': 'CeNGEN Barrett bulk RNAseq, within-neuron-type CV, gene IDs via mygene.info',
    'wiring_group': {'n_resolved_with_data': len(wiring_cv), 'median_cv': float(np.median(wv)) if wv else None,
                     'per_gene': {g: round(c, 3) for g, c in sorted(wiring_cv.items(), key=lambda x: x[1])}},
    'channel_group': {'n_resolved_with_data': len(cv), 'median_cv': float(np.median(cv)) if cv else None,
                      'per_gene': {g: round(c, 3) for g, c in sorted(channel_cv.items(), key=lambda x: x[1])}},
    'mannwhitney_channel_greater': {'U': float(U), 'p': float(p),
                                    'median_channel': float(np.median(cv)), 'median_wiring': float(np.median(wv)),
                                    'n_channel': len(cv), 'n_wiring': len(wv)},
    'caveats': ['group-level test (robust to individual gene); within-type CV proxy for cell-to-cell '
                'regulatory variability; gene-group membership hand-curated from biophysical class.'],
}
json.dump(out, open('/root/autodl-tmp/paper2_BV2_group_cv.json', 'w'), indent=2)
print('=== BV2 group-level expression variability ===', flush=True)
print(f'WIRING group (stiff): n={len(wv)} median CV={np.median(wv):.3f}', flush=True)
print(f'CHANNEL group (sloppy): n={len(cv)} median CV={np.median(cv):.3f}', flush=True)
print(f'MannWhitney channel>wiring: U={U:.0f} p={p:.4f}', flush=True)
print('wiring per-gene:', {g: round(c, 3) for g, c in sorted(wiring_cv.items(), key=lambda x: x[1])}, flush=True)
print('channel per-gene:', {g: round(c, 3) for g, c in sorted(channel_cv.items(), key=lambda x: x[1])}, flush=True)
print('BV2_DONE', flush=True)
