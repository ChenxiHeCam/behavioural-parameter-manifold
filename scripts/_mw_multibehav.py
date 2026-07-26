"""MB: is the low-dim manifold intrinsic to kinematics, or a one-behaviour artifact?
Run modWorm under 4 distinct stimuli/behaviours (gentle/harsh x anterior/posterior touch ->
reversal vs forward escape). Per behaviour: stiff subspace (Hessian eigvecs) + eff-dim. Then ask:
do different behaviours SHARE the stiff subspace (low-dim is intrinsic, 'eating==boxing') or have
DISTINCT stiff subspaces (each behaviour low-dim but the repertoire is higher-dim)?
Also: eff-dim of the UNION of all behaviours."""
import os, sys, json, subprocess, time, math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.linalg import subspace_angles

PY = '/root/miniconda3/bin/python'; W = '/root/autodl-tmp/mw_behav_worker.py'
NSTEP = 150; DELTA = 0.25; CONC = 12; KSS = 2
MECH = ['gap', 'syn', 'leak', 'Cm', 'rise', 'fall', 'B']
OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'rms_heading', 'final_curve']
STIMS = {'gentle_ant': 'input_mat_gentle_ant_touch', 'gentle_post': 'input_mat_gentle_post_touch',
         'harsh_ant': 'input_mat_harsh_ant_touch', 'harsh_post': 'input_mat_harsh_post_touch'}


def run(stim, scale):
    sp = dict(scale); sp['n_steps'] = NSTEP; sp['stim'] = stim
    try:
        p = subprocess.run([PY, W, json.dumps(sp)], capture_output=True, text=True, timeout=1800)
        for ln in p.stdout.splitlines():
            if ln.startswith('RESULT'):
                return json.loads(ln[7:])['obs']
    except Exception:
        pass
    return None


jobs = []
for bn in STIMS:
    jobs.append((bn, 'base', {}))
    for m in MECH:
        jobs.append((bn, m + '+', {m: 1 + DELTA})); jobs.append((bn, m + '-', {m: 1 - DELTA}))
t0 = time.time()
print(f'{len(jobs)} rollouts ...', flush=True)
res = {}


def do(j):
    bn, tag, sc = j; return (bn, tag), run(STIMS[bn], sc)


with ThreadPoolExecutor(max_workers=CONC) as ex:
    for k, o in ex.map(do, jobs):
        res[k] = o
print(f'rollouts done {time.time()-t0:.0f}s', flush=True)

dln = math.log(1 + DELTA) - math.log(1 - DELTA)
beh = {}
allJ = []
for bn in STIMS:
    base = res.get((bn, 'base'))
    if base is None:
        continue
    scale = np.array([abs(base[k]) + 1e-9 for k in OBS])
    J = np.zeros((len(OBS), len(MECH))); ok = True
    for j, m in enumerate(MECH):
        op, om = res.get((bn, m + '+')), res.get((bn, m + '-'))
        if op is None or om is None:
            ok = False; break
        J[:, j] = ((np.array([op[k] for k in OBS]) - np.array([om[k] for k in OBS])) / scale) / dln
    if not ok:
        continue
    H = J.T @ J
    w, V = np.linalg.eigh(H); idx = np.argsort(w)[::-1]; w = w[idx]; V = V[:, idx]
    cs = np.cumsum(np.maximum(w, 0)) / max(np.maximum(w, 0).sum(), 1e-30)
    beh[bn] = {'eff_dim_90': int(np.searchsorted(cs, 0.9) + 1), 'eff_dim_99': int(np.searchsorted(cs, 0.99) + 1),
               'top_eigvec': [round(float(v), 3) for v in V[:, 0]], 'V': V, 'net_disp': base['net_disp']}
    allJ.append(J)

# cross-behaviour stiff-subspace alignment (top-KSS eigvecs); cos=1 shared, 0 distinct
names = list(beh)
align = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = beh[names[i]]['V'][:, :KSS]; b = beh[names[j]]['V'][:, :KSS]
        align[f'{names[i]}__{names[j]}'] = round(float(np.cos(subspace_angles(a, b)).mean()), 3)

# union eff-dim
Jc = np.vstack(allJ); wc = np.sort(np.maximum(np.linalg.eigvalsh(Jc.T @ Jc), 0))[::-1]
csc = np.cumsum(wc) / max(wc.sum(), 1e-30)
union90 = int(np.searchsorted(csc, 0.9) + 1); union99 = int(np.searchsorted(csc, 0.99) + 1)
mean_align = float(np.mean(list(align.values()))) if align else None

out = {
    'experiment': 'MB_behaviour_specificity_of_stiff_subspace (REAL modWorm, 4 behaviours)',
    'question': 'is low-dim intrinsic (shared stiff subspace across behaviours) or one-behaviour artifact',
    'per_behaviour_eff_dim': {bn: [beh[bn]['eff_dim_90'], beh[bn]['eff_dim_99'], 'of 7',
                                   'net_disp=%.3f' % beh[bn]['net_disp']] for bn in beh},
    'per_behaviour_top_stiff_eigvec': {bn: dict(zip(MECH, beh[bn]['top_eigvec'])) for bn in beh},
    'cross_behaviour_stiff_subspace_alignment_cos': align,
    'mean_cross_behaviour_alignment': mean_align,
    'union_eff_dim_90_99': [union90, union99],
    'verdict': ('shared low-dim subspace across behaviours (intrinsic)' if mean_align and mean_align > 0.7
                else 'behaviour-specific stiff subspaces (repertoire higher-dim than any one behaviour)'
                if mean_align is not None else 'inconclusive'),
    'elapsed_sec': round(time.time() - t0, 1),
}
json.dump(out, open('/root/autodl-tmp/MB_behaviour_specificity.json', 'w'), indent=2)
print('=== MB behaviour-specificity ===', flush=True)
for bn in beh:
    print(f'  {bn}: eff-dim {beh[bn]["eff_dim_90"]}/{beh[bn]["eff_dim_99"]} of 7, net_disp {beh[bn]["net_disp"]:.3f}', flush=True)
print('cross-behaviour stiff-subspace alignment cos:', align, flush=True)
print('mean alignment %.3f | union eff-dim %d/%d | verdict: %s' % (mean_align or -1, union90, union99, out['verdict']), flush=True)
print('MB_DONE %.0fs' % (time.time() - t0), flush=True)
