"""Paper2 G1: real mutant behavioural identifiability vs model B3 prediction.
Tests whether the model's per-channel behavioural-detectability ranking (B3 signature_norm)
predicts which real C. elegans channel mutants have weak vs strong behavioural phenotypes.
Data: OWMD (Zenodo community open-worm-movement-database) Tierpsy/segworm features_means
(~700 named features per worm). Streamed in-memory, features_means extracted, file discarded.
Prediction: fast Ca channels (egl-19, unc-2; low signature_norm) -> small behavioural distance
to N2 (behavioural screens MISS them); synaptic / Na-leak / gap (high signature_norm) -> large.
"""
import urllib.request, json, h5py, io, numpy as np, time, sys
from scipy import stats
from concurrent.futures import ThreadPoolExecutor

COM = 'open-worm-movement-database'
OUT = '/root/autodl-tmp/paper2_G1_mutant_identifiability.json'
KWORM = 12          # worms per mutant gene
KN2 = 40            # N2 reference worms
TIMEOUT = 120
NWORKERS = 12       # parallel downloads (China->Zenodo ~0.4 MB/s per connection)

# model per-channel behavioural detectability (B3 signature_norm), only channels with OWMD data
MODEL_SIG = {'egl-19': 0.0685, 'unc-2': 0.1179, 'egl-2': 0.7081,
             'nca': 0.8856, 'gj': 0.3820, 'syn': 1.1899}
GENE2CH = {'egl-19': 'egl-19', 'unc-2': 'unc-2', 'egl-2': 'egl-2',
           'nca-1': 'nca', 'nca-2': 'nca', 'unc-77': 'nca', 'unc-79': 'nca',
           'unc-9': 'gj', 'unc-7': 'gj',
           'unc-17': 'syn', 'unc-29': 'syn', 'unc-38': 'syn'}
CH_CLASS = {'egl-19': 'fast_Ca', 'unc-2': 'fast_Ca', 'egl-2': 'K_channel',
            'nca': 'Na_leak', 'gj': 'gap', 'syn': 'synaptic'}
DROP = {'worm_index', 'n_frames', 'n_valid_skel', 'first_frame'}


def api(term, size=25, page=1):
    # Zenodo unauthenticated API caps size at ~25 (size>=40 -> HTTP 400); paginate.
    url = f'https://zenodo.org/api/records?q=communities%3A{COM}%20AND%20%22{term}%22&size={size}&page={page}'
    last = None
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'g1'}), timeout=90))
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def feat_links(term, k):
    out, page = [], 1
    while len(out) < k and page <= 10:
        hits = api(term, size=25, page=page).get('hits', {}).get('hits', [])
        if not hits:
            break
        for h in hits:
            ff = [f for f in h.get('files', []) if f.get('key', '').lower().endswith('_features.hdf5')]
            if ff:
                l = ff[0].get('links', {}).get('self') or ff[0].get('links', {}).get('download')
                if l:
                    out.append(l)
            if len(out) >= k:
                break
        page += 1
    return out


def load_means(link):
    raw = None
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(link, headers={'User-Agent': 'g1'}), timeout=TIMEOUT).read()
            break
        except Exception:
            time.sleep(2 * (attempt + 1))
    if raw is None:
        raise IOError('download failed')
    with h5py.File(io.BytesIO(raw), 'r') as hf:
        fm = hf['features_means'][:]
    return {n: float(fm[n][0]) for n in fm.dtype.names if n not in DROP}


t0 = time.time()
# 1. gather links per target (serial API pagination; small JSON)
targets = [('N2', KN2)] + [(g, KWORM) for g in GENE2CH]
jobs = []   # (label, link)
for label, k in targets:
    links = feat_links(label, k)
    print(f'links {label}: {len(links)}', flush=True)
    jobs += [(label, l) for l in links]
print(f'total files to download: {len(jobs)} (~{len(jobs)*16}MB)', flush=True)

# 2. parallel download + extract features_means
done = [0]


def _fetch(job):
    label, link = job
    try:
        r = load_means(link)
        done[0] += 1
        if done[0] % 20 == 0:
            print(f'  downloaded {done[0]}/{len(jobs)} ({time.time()-t0:.0f}s)', flush=True)
        return label, r
    except Exception:
        return label, None


buckets = {}
with ThreadPoolExecutor(max_workers=NWORKERS) as ex:
    for label, r in ex.map(_fetch, jobs):
        if r is not None:
            buckets.setdefault(label, []).append(r)

n2_rows = buckets.get('N2', [])
print('N2 worms:', len(n2_rows), flush=True)
mut_rows = {g: buckets[g] for g in GENE2CH if buckets.get(g)}
for g in GENE2CH:
    print(f'{g}: {len(buckets.get(g, []))} worms', flush=True)

# common feature set across all worms
allrows = n2_rows + [r for rs in mut_rows.values() for r in rs]
common = set(allrows[0])
for r in allrows:
    common.intersection_update(r.keys())
common = sorted(common)
print('common features:', len(common), flush=True)


def mat(rows):
    return np.array([[r[f] for f in common] for r in rows], float)


N2 = mat(n2_rows)
med = np.nanmedian(N2, axis=0)
mad = stats.median_abs_deviation(N2, axis=0, scale='normal', nan_policy='omit')
keep = mad > 0


def dist_to_n2(X):
    m = np.nanmedian(X, axis=0)
    z = (m[keep] - med[keep]) / mad[keep]
    z = z[np.isfinite(z)]
    return float(np.sqrt(np.mean(z ** 2)))


gene_dist = {g: dist_to_n2(mat(r)) for g, r in mut_rows.items()}
gene_nworm = {g: len(r) for g, r in mut_rows.items()}

# per-channel aggregate (median over genes mapping to channel)
ch_dist = {}
for ch in set(GENE2CH[g] for g in mut_rows):
    ds = [gene_dist[g] for g in mut_rows if GENE2CH[g] == ch]
    ch_dist[ch] = float(np.median(ds))

chs = [c for c in ch_dist if c in MODEL_SIG]
xs = [MODEL_SIG[c] for c in chs]
ys = [ch_dist[c] for c in chs]
rho, p = stats.spearmanr(xs, ys)

# class contrast: fast Ca (predicted sloppy/weak) vs the rest (predicted detectable)
fastca = [gene_dist[g] for g in mut_rows if CH_CLASS[GENE2CH[g]] == 'fast_Ca']
rest = [gene_dist[g] for g in mut_rows if CH_CLASS[GENE2CH[g]] != 'fast_Ca']
if fastca and rest:
    U, pmw = stats.mannwhitneyu(rest, fastca, alternative='greater')
else:
    U, pmw = None, None

result = {
    'experiment': 'G1_real_mutant_behavioural_identifiability',
    'source': 'OWMD (Zenodo community open-worm-movement-database), Tierpsy/segworm features_means',
    'n_features_common': len(common),
    'n_features_used_nonzero_mad': int(keep.sum()),
    'model_prediction': 'B3 per-channel signature_norm (behavioural detectability of a knockdown)',
    'model_signature_norm': MODEL_SIG,
    'gene_to_channel': GENE2CH,
    'gene_behavioural_distance_to_N2': gene_dist,
    'gene_n_worms': gene_nworm,
    'n_N2_worms': len(n2_rows),
    'channel_behavioural_distance': ch_dist,
    'per_channel_correlation': {
        'channels': chs, 'model_signature': xs, 'real_distance': ys,
        'spearman_rho': float(rho), 'spearman_p': float(p), 'n': len(chs),
    },
    'class_contrast_fastCa_vs_rest': {
        'hypothesis': 'fast Ca (egl-19,unc-2) behaviourally weaker than synaptic/Na-leak/gap',
        'median_fastCa': float(np.median(fastca)) if fastca else None,
        'median_rest': float(np.median(rest)) if rest else None,
        'mannwhitney_U': float(U) if U is not None else None,
        'mannwhitney_p_rest_greater': float(pmw) if pmw is not None else None,
        'n_fastCa': len(fastca), 'n_rest': len(rest),
    },
    'caveats': [
        'allele heterogeneity: OWMD strains are specific alleles (hypomorph vs null), not matched knockdown levels.',
        'gene->model-channel map is hand-curated; aggregate channels (syn/gj/nca) map to multiple genes.',
        'observational; behavioural distance is robust-z RMS to N2 in segworm feature space.',
        'small n at channel level (n=%d).' % len(chs),
    ],
    'elapsed_sec': round(time.time() - t0, 1),
}
json.dump(result, open(OUT, 'w'), indent=2)
print('=== G1 RESULT ===', flush=True)
print('per-channel Spearman rho=%.3f p=%.4f (n=%d)' % (rho, p, len(chs)), flush=True)
print('channels:', list(zip(chs, [round(x, 3) for x in xs], [round(y, 2) for y in ys])), flush=True)
print('class contrast: med_fastCa=%.2f med_rest=%.2f MW p(rest>fastCa)=%s' %
      (np.median(fastca) if fastca else float('nan'),
       np.median(rest) if rest else float('nan'), pmw), flush=True)
print('gene distances:', {g: round(v, 2) for g, v in sorted(gene_dist.items(), key=lambda kv: kv[1])}, flush=True)
print('G1_DONE elapsed=%.0fs' % (time.time() - t0), flush=True)
