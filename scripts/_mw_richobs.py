"""RICH: is the low eff-dim REAL, or capped by using only 6 coarse observables?
eff-dim <= min(#observables, #params). With 6 scalar observables, eff-dim <= 6 BY CONSTRUCTION.
Re-measure the per-mechanism eff-dim using the FULL posture trajectory (T x 48 tangent angles =
thousands of observables) instead of 6 scalars, at several durations. If eff-dim grows toward 7,
the earlier low value was an observable artifact; if it stays low, the low-dim is real.
Uses mw_posture_worker (outputs body tangent-angle matrix)."""
import os, sys, json, subprocess, time, gzip, base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor

PY = '/root/miniconda3/envs/flygym_fresh/bin/python'; W = '/root/autodl-tmp/mw_posture_worker.py'
DELTA = 0.25; L = 48
MECH = ['gap', 'syn', 'leak', 'Cm', 'rise', 'fall', 'B']


def run_post(scale, nstep):
    sp = dict(scale); sp['n_steps'] = nstep; sp['L'] = L
    try:
        p = subprocess.run([PY, W, json.dumps(sp)], capture_output=True, text=True, timeout=2400)
        for ln in p.stdout.splitlines():
            if ln.startswith('RESULT'):
                d = json.loads(ln[7:])
                return np.frombuffer(gzip.decompress(base64.b64decode(d['b64'])), np.float32).reshape(d['shape'])
    except Exception:
        pass
    return None


def eff_dim(w):
    w = np.maximum(w, 0); w = w[w > 0]
    cs = np.cumsum(np.sort(w)[::-1]) / w.sum()
    return int(np.searchsorted(cs, 0.9) + 1), int(np.searchsorted(cs, 0.99) + 1), [round(float(x), 4) for x in np.sort(w)[::-1]]


t0 = time.time()
results = {}
for NSTEP in [150, 300]:
    # base + 14 perturbations, full posture each
    jobs = {'base': {}}
    for m in MECH:
        jobs[m + '+'] = {m: 1 + DELTA}; jobs[m + '-'] = {m: 1 - DELTA}
    post = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {k: ex.submit(run_post, v, NSTEP) for k, v in jobs.items()}
        for k, f in futs.items():
            post[k] = f.result()
    base = post['base']
    if base is None:
        continue
    Tmin = min(p.shape[0] for p in post.values() if p is not None)
    # rich Jacobian: columns = flattened posture-trajectory response per mechanism
    cols = []
    for m in MECH:
        op, om = post.get(m + '+'), post.get(m + '-')
        if op is None or om is None:
            cols.append(np.zeros(Tmin * L)); continue
        cols.append(((op[:Tmin] - om[:Tmin]) / (2 * DELTA)).astype(float).flatten())
    J = np.array(cols).T                      # (Tmin*L, 7)
    # whiten by per-observable std across the posture (so big-amplitude points don't dominate)
    scale = base[:Tmin].std(0).clip(1e-6)     # (L,)
    scale_full = np.tile(scale, Tmin)
    Jw = J / scale_full[:, None]
    H = Jw.T @ Jw
    e90, e99, spec = eff_dim(np.linalg.eigvalsh(H))
    results[f'NSTEP{NSTEP}'] = {'n_observables': int(Tmin * L), 'eff_dim_90': e90, 'eff_dim_99': e99,
                               'spectrum_normalised': [round(s / spec[0], 4) for s in spec] if spec[0] > 0 else spec,
                               'T': int(Tmin)}
    print(f'NSTEP={NSTEP}: {Tmin*L} observables -> eff-dim {e90}/{e99} of 7 (vs 6-scalar ~1-3)', flush=True)
    print('  normalised spectrum:', [round(s / spec[0], 4) for s in spec] if spec[0] > 0 else spec, flush=True)

out = {
    'experiment': 'RICH_observable_effdim (REAL modWorm): is low-dim real or a 6-observable artifact',
    'method': 'per-mechanism eff-dim using FULL posture trajectory (T x 48 tangent angles) as observables, vs the 6 coarse scalars',
    'results_by_duration': results,
    'comparison': 'the 6-scalar single-behaviour eff-dim was ~1-3 of 7 (capped by 6 observables)',
    'elapsed_sec': round(time.time() - t0, 1),
}
json.dump(out, open('/root/autodl-tmp/RICH_observable.json', 'w'), indent=2)
print('=== RICH observable eff-dim ===', flush=True)
for k, v in results.items():
    print(f'  {k}: {v["n_observables"]} obs -> eff-dim {v["eff_dim_90"]}/{v["eff_dim_99"]} of 7', flush=True)
print('RICH_DONE %.0fs' % (time.time() - t0), flush=True)
