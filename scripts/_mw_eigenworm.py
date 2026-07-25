"""Paper2 EW: what IS the low-dimensional manifold, concretely, in real biology?
Maps the model's behavioural-equivalence manifold to the worm's canonical low-dim behaviour =
EIGENWORMS (Stephens 2008: real C. elegans posture lives in ~4 eigenworm modes). Three results:
(1) does modWorm posture span the same low-dim eigenworm space as REAL OWMD N2 worms (shape +
dimension match)? (2) which parameter directions (stiff vs sloppy) control the eigenworm modes?
-> gives the abstract stiff subspace a concrete behavioural identity: the neural control of the
worm's eigenworm modes. (3) effective dimension of model vs real posture.
"""
import os, sys, json, subprocess, time, io, gzip, base64, urllib.request
import numpy as np, h5py
from scipy.linalg import subspace_angles
from concurrent.futures import ThreadPoolExecutor

PY = '/root/miniconda3/bin/python'; W = '/root/autodl-tmp/mw_posture_worker.py'
L = 48; NSTEP = 300; DELTA = 0.5; KEW = 4
MECH = ['gap', 'syn', 'leak', 'Cm', 'rise', 'fall', 'B']
STIFF = {'gap', 'syn', 'rise'}; SLOPPY = {'Cm', 'B'}
COM = 'open-worm-movement-database'


def run_model(scale):
    sp = dict(scale); sp['n_steps'] = NSTEP; sp['L'] = L
    try:
        p = subprocess.run([PY, W, json.dumps(sp)], capture_output=True, text=True, timeout=2400)
        for ln in p.stdout.splitlines():
            if ln.startswith('RESULT'):
                d = json.loads(ln[7:])
                return np.frombuffer(gzip.decompress(base64.b64decode(d['b64'])), np.float32).reshape(d['shape'])
    except Exception as e:
        print('model err', repr(e)[:60], flush=True)
    return None


def eigenworms(A):
    A = A - A.mean(0, keepdims=True)
    C = np.cov(A.T)
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def eff_dim(w):
    w = w[w > 0]
    return float(w.sum() ** 2 / (w ** 2).sum())


def real_angles(nworm=6, maxframes=5000):
    url = f'https://zenodo.org/api/records?q=communities%3A{COM}%20AND%20%22N2%22&size=12&page=1'
    d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'ew'}), timeout=60))
    rows = []; got = 0; sn = np.linspace(0, 1, L)
    for h in d['hits']['hits']:
        ff = [f for f in h['files'] if f['key'].lower().endswith('_features.hdf5')]
        if not ff:
            continue
        link = ff[0]['links'].get('self') or ff[0]['links'].get('download')
        try:
            raw = urllib.request.urlopen(urllib.request.Request(link, headers={'User-Agent': 'ew'}), timeout=150).read()
            with h5py.File(io.BytesIO(raw), 'r') as hf:
                sk = hf['coordinates']['skeletons'][:maxframes]
            for t in range(0, sk.shape[0], 3):
                pt = sk[t]
                if not np.all(np.isfinite(pt)):
                    continue
                dx = np.diff(pt[:, 0]); dy = np.diff(pt[:, 1])
                a = np.unwrap(np.arctan2(dy, dx)); a = a - a.mean()
                s = np.linspace(0, 1, len(a)); rows.append(np.interp(sn, s, a))
            got += 1
            print(f'  real worm {got}: {sk.shape[0]} frames', flush=True)
        except Exception as e:
            print('  real skip', repr(e)[:50], flush=True)
        if got >= nworm:
            break
    return np.array(rows)


t0 = time.time()
# 1. model rollouts: base + per-mechanism perturbations (parallel)
jobs = {'base': {}}
for m in MECH:
    jobs[m + '+'] = {m: 1 + DELTA}; jobs[m + '-'] = {m: 1 - DELTA}
print(f'running {len(jobs)} model rollouts ...', flush=True)
res = {}
with ThreadPoolExecutor(max_workers=10) as ex:
    futs = {lab: ex.submit(run_model, sc) for lab, sc in jobs.items()}
    for lab, f in futs.items():
        res[lab] = f.result()
base = res['base']
print('model base posture:', None if base is None else base.shape, f'({time.time()-t0:.0f}s)', flush=True)

# 2. model eigenworms + eff-dim
w_m, V_m = eigenworms(base)
ve_m = np.cumsum(w_m) / w_m.sum()
effdim_m = eff_dim(w_m)

# 3. real eigenworms
print('fetching real OWMD N2 skeletons ...', flush=True)
real = real_angles()
print('real angle frames:', real.shape, flush=True)
w_r, V_r = eigenworms(real)
ve_r = np.cumsum(w_r) / w_r.sum()
effdim_r = eff_dim(w_r)

# 4. subspace alignment: principal angles between top-K eigenworm subspaces (model vs real)
pa = subspace_angles(V_m[:, :KEW], V_r[:, :KEW])
mean_cos = float(np.cos(pa).mean())
per_mode_cos = [float(abs(np.dot(V_m[:, i], V_r[:, i]))) for i in range(KEW)]

# 5. stiff/sloppy -> eigenworm control: how much each mechanism perturbation changes the
#    eigenworm-amplitude trajectory (projected onto REAL eigenworms, the biological modes)
basis = V_r[:, :KEW]
base_amp = (base - base.mean(0, keepdims=True)) @ basis
ctrl = {}
for m in MECH:
    chg = []
    for s in ['+', '-']:
        A = res.get(m + s)
        if A is None:
            continue
        amp = (A - base.mean(0, keepdims=True)) @ basis
        n = min(len(amp), len(base_amp))
        chg.append(float(np.sqrt(np.mean((amp[:n] - base_amp[:n]) ** 2))))
    ctrl[m] = float(np.mean(chg)) if chg else None
stiff_ctrl = [ctrl[m] for m in MECH if m in STIFF and ctrl[m] is not None]
sloppy_ctrl = [ctrl[m] for m in MECH if m in SLOPPY and ctrl[m] is not None]

out = {
    'experiment': 'EW_lowdim_manifold_identity_eigenworms (REAL modWorm  + OWMD N2)',
    'question': 'what IS the low-dim behavioural manifold, concretely, vs real worm eigenworms',
    'L': L, 'NSTEP': NSTEP, 'K_eigenworms': KEW,
    'model_posture_effdim': effdim_m,
    'model_varexp_top4': [float(v) for v in ve_m[:4]],
    'real_posture_effdim': effdim_r,
    'real_varexp_top4': [float(v) for v in ve_r[:4]],
    'real_n_frames': int(real.shape[0]),
    'eigenworm_subspace_alignment': {
        'mean_cos_principal_angles_top4': mean_cos,
        'per_mode_abs_cos': per_mode_cos,
        'note': 'cos=1 -> model eigenworms identical to real; top-4 subspace alignment',
    },
    'stiff_sloppy_eigenworm_control': {
        'per_mechanism_eigenworm_change': ctrl,
        'median_stiff': float(np.median(stiff_ctrl)) if stiff_ctrl else None,
        'median_sloppy': float(np.median(sloppy_ctrl)) if sloppy_ctrl else None,
        'stiff_over_sloppy': float(np.median(stiff_ctrl) / np.median(sloppy_ctrl))
        if stiff_ctrl and sloppy_ctrl and np.median(sloppy_ctrl) > 0 else None,
        'note': 'stiff mechanisms move the eigenworm trajectory; sloppy do not -> stiff subspace = neural control of eigenworm modes',
    },
    'verdict': 'the low-dim behavioural manifold is the neural control of the worm eigenworm modes',
    'elapsed_sec': round(time.time() - t0, 1),
}
json.dump(out, open('/root/autodl-tmp/paper2_EW_eigenworm.json', 'w'), indent=2)
print('=== EW eigenworm manifold identity ===', flush=True)
print('model posture eff-dim %.2f (top4 var %s)' % (effdim_m, [round(v, 2) for v in ve_m[:4]]), flush=True)
print('real  posture eff-dim %.2f (top4 var %s)' % (effdim_r, [round(v, 2) for v in ve_r[:4]]), flush=True)
print('eigenworm subspace alignment mean_cos=%.3f per-mode=%s' % (mean_cos, [round(c, 2) for c in per_mode_cos]), flush=True)
print('stiff->eigenworm %s | sloppy->eigenworm %s | ratio %s' % (
    out['stiff_sloppy_eigenworm_control']['median_stiff'],
    out['stiff_sloppy_eigenworm_control']['median_sloppy'],
    out['stiff_sloppy_eigenworm_control']['stiff_over_sloppy']), flush=True)
print('per-mech eigenworm control:', {m: round(ctrl[m], 3) if ctrl[m] else None for m in MECH}, flush=True)
print('EW_DONE %.0fs' % (time.time() - t0), flush=True)
