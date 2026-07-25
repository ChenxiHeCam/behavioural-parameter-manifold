"""Paper2 SAT: does the identifiable (stiff) subspace SATURATE as more behaviours constrain it?
The user's question: with enough behaviours, do we pin down a higher-dim subspace (fewer free DOF)?
Run modWorm under N distinct behaviours (4 touch presets x stimulus amplitudes). For each, the
observable Jacobian J_b (6 obs x 7 mech). The UNION stiff subspace = eff-dim of the stacked
Jacobians. Plot eff-dim vs number-of-behaviours (averaged over random orderings): does it saturate
below 7? The saturation dimension = max behaviour can ever identify; the residual = truly degenerate
(unconstrainable by ANY behaviour) = the irreducible biological neutral space.
"""
import os, sys, json, subprocess, time, math, itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor

PY = '/root/miniconda3/envs/flygym_fresh/bin/python'; W = '/root/autodl-tmp/mw_behav_worker.py'
NSTEP = 150; DELTA = 0.25; CONC = 12
MECH = ['gap', 'syn', 'leak', 'Cm', 'rise', 'fall', 'B']
OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'rms_heading', 'final_curve']
PRESETS = ['input_mat_gentle_ant_touch', 'input_mat_gentle_post_touch',
           'input_mat_harsh_ant_touch', 'input_mat_harsh_post_touch']
SCALES = [0.5, 1.0, 2.0]
BEHAVIOURS = [(p, s) for p in PRESETS for s in SCALES]   # 12 behaviours


def run(stim, sscale, scale):
    sp = dict(scale); sp['n_steps'] = NSTEP; sp['stim'] = stim; sp['stim_scale'] = sscale
    try:
        p = subprocess.run([PY, W, json.dumps(sp)], capture_output=True, text=True, timeout=1800)
        for ln in p.stdout.splitlines():
            if ln.startswith('RESULT'):
                return json.loads(ln[7:])['obs']
    except Exception:
        pass
    return None


def eff_dim(w):
    w = np.maximum(w, 0); w = w[w > 0]
    cs = np.cumsum(np.sort(w)[::-1]) / w.sum()
    return int(np.searchsorted(cs, 0.9) + 1), int(np.searchsorted(cs, 0.99) + 1)


t0 = time.time()
jobs = []
for bi, (p, s) in enumerate(BEHAVIOURS):
    jobs.append((bi, 'base', {}))
    for m in MECH:
        jobs.append((bi, m + '+', {m: 1 + DELTA})); jobs.append((bi, m + '-', {m: 1 - DELTA}))
print(f'{len(jobs)} rollouts over {len(BEHAVIOURS)} behaviours ...', flush=True)
res = {}
cnt = [0]


def do(j):
    bi, tag, sc = j; p, s = BEHAVIOURS[bi]; o = run(p, s, sc); cnt[0] += 1
    if cnt[0] % 30 == 0:
        print(f'  {cnt[0]}/{len(jobs)} ({time.time()-t0:.0f}s)', flush=True)
    return (bi, tag), o


with ThreadPoolExecutor(max_workers=CONC) as ex:
    for k, o in ex.map(do, jobs):
        res[k] = o

# per-behaviour Jacobians
dln = math.log(1 + DELTA) - math.log(1 - DELTA)
Js = []; valid = []
for bi in range(len(BEHAVIOURS)):
    base = res.get((bi, 'base'))
    if base is None:
        continue
    scale = np.array([abs(base[k]) + 1e-9 for k in OBS])
    J = np.zeros((len(OBS), len(MECH))); ok = True
    for j, m in enumerate(MECH):
        op, om = res.get((bi, m + '+')), res.get((bi, m + '-'))
        if op is None or om is None:
            ok = False; break
        J[:, j] = ((np.array([op[k] for k in OBS]) - np.array([om[k] for k in OBS])) / scale) / dln
    if ok:
        Js.append(J); valid.append(bi)
nB = len(Js)
print(f'{nB} valid behaviours', flush=True)

# saturation curve: eff-dim of union of k behaviours, averaged over random orderings
rng = np.random.default_rng(0)
curve = []
for k in range(1, nB + 1):
    e90s, e99s = [], []
    n_orders = min(20, math.comb(nB, k)) if k < nB else 1
    seen = set()
    for _ in range(n_orders * 3):
        if len(seen) >= n_orders:
            break
        sub = tuple(sorted(rng.choice(nB, size=k, replace=False)))
        if sub in seen:
            continue
        seen.add(sub)
        Jc = np.vstack([Js[i] for i in sub])
        w = np.linalg.eigvalsh(Jc.T @ Jc)
        e90, e99 = eff_dim(w)
        e90s.append(e90); e99s.append(e99)
    curve.append({'n_behaviours': k, 'eff_dim_90_mean': float(np.mean(e90s)), 'eff_dim_99_mean': float(np.mean(e99s)),
                  'eff_dim_99_max': int(np.max(e99s))})

# full-union spectrum
Jall = np.vstack(Js); wall = np.sort(np.maximum(np.linalg.eigvalsh(Jall.T @ Jall), 0))[::-1]
csall = np.cumsum(wall) / wall.sum()
union90, union99 = eff_dim(wall)
# residual truly-sloppy: # eigvals below 1e-3 of max (unconstrainable by ANY of these behaviours)
residual = int(np.sum(wall < wall.max() * 1e-3))

out = {
    'experiment': 'SAT_identifiability_saturation (REAL modWorm, %d behaviours)' % nB,
    'question': 'does the identifiable stiff subspace saturate as more behaviours constrain it',
    'n_behaviours': nB, 'n_params': len(MECH),
    'saturation_curve': curve,
    'full_union_eff_dim_90_99': [union90, union99],
    'full_union_spectrum': [round(float(x), 4) for x in wall],
    'truly_degenerate_dims_below_1e-3': residual,
    'verdict': 'identifiable subspace saturates at eff-dim %d/%d; %d direction(s) remain truly degenerate '
               '(unconstrainable by any of these behaviours = irreducible neutral space)' % (union99, len(MECH), residual),
    'elapsed_sec': round(time.time() - t0, 1),
}
json.dump(out, open('/root/autodl-tmp/paper2_SAT_saturation.json', 'w'), indent=2)
print('=== SAT identifiability saturation ===', flush=True)
for c in curve:
    print(f'  {c["n_behaviours"]} behaviours -> eff-dim(99) {c["eff_dim_99_mean"]:.2f} (max {c["eff_dim_99_max"]})', flush=True)
print('full union eff-dim %d/%d of 7 | truly degenerate dims %d' % (union90, union99, residual), flush=True)
print('union spectrum:', [round(float(x), 3) for x in wall], flush=True)
print('SAT_DONE %.0fs' % (time.time() - t0), flush=True)
