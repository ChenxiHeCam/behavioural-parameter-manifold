"""CP: is sloppy = UNUSED parameters, or COUPLED/degenerate combinations?
Extract the behavioural-Hessian EIGENVECTORS (combinations of the 7 mechanisms). Then move the
parameters ALONG a stiff eigenvector vs a sloppy eigenvector (coordinated multi-parameter moves of
matched magnitude) and along a single parameter. If sloppy = coupled degeneracy: moving along the
sloppy eigenvector (coordinated) preserves behaviour, while moving a single parameter or along the
stiff eigenvector changes it. Demonstrates degeneracy is COORDINATED-COMBINATION, not unused params.
"""
import os, sys, json, subprocess, time, math
import numpy as np
from concurrent.futures import ThreadPoolExecutor

PY = '/root/miniconda3/bin/python'; W = '/root/autodl-tmp/mw_real_worker.py'
NSTEP = 150; DELTA = 0.25
MECH = ['gap', 'syn', 'leak', 'Cm', 'rise', 'fall', 'B']
OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'rms_heading', 'final_curve']


def run(scale):
    sp = dict(scale); sp['n_steps'] = NSTEP
    try:
        p = subprocess.run([PY, W, json.dumps(sp)], capture_output=True, text=True, timeout=1800)
        for ln in p.stdout.splitlines():
            if ln.startswith('RESULT'):
                return json.loads(ln[7:])['obs']
    except Exception:
        pass
    return None


t0 = time.time()
# Phase 1: Jacobian -> Hessian -> eigenvectors
jobs = {'base': {}}
for m in MECH:
    jobs[m + '+'] = {m: 1 + DELTA}; jobs[m + '-'] = {m: 1 - DELTA}
res = {}
with ThreadPoolExecutor(max_workers=12) as ex:
    futs = {k: ex.submit(run, v) for k, v in jobs.items()}
    for k, f in futs.items():
        res[k] = f.result()
base = res['base']
scale = np.array([abs(base[k]) + 1e-9 for k in OBS]); dln = math.log(1 + DELTA) - math.log(1 - DELTA)
J = np.zeros((6, 7))
for j, m in enumerate(MECH):
    op, om = res[m + '+'], res[m + '-']
    J[:, j] = ((np.array([op[k] for k in OBS]) - np.array([om[k] for k in OBS])) / scale) / dln
H = J.T @ J
w, V = np.linalg.eigh(H); idx = np.argsort(w)[::-1]; w = w[idx]; V = V[:, idx]
v_stiff = V[:, 0]; v_sloppy = V[:, -1]
# "is the stiff direction a COMBINATION?" -> participation ratio of its squared loadings
def participation(v):
    p = v ** 2 / (v ** 2).sum()
    return float(1.0 / (p ** 2).sum())   # 1=single param, 7=all equal


def behav_dist(o):
    return float(np.sqrt(np.mean(((np.array([o[k] for k in OBS]) - np.array([base[k] for k in OBS])) / scale) ** 2)))


# Phase 2: move along eigenvectors (coordinated) vs single param, matched magnitude alpha
moves = {}
for alpha in [0.3, 0.6]:
    for name, v in [('stiff_eigvec', v_stiff), ('sloppy_eigvec', v_sloppy)]:
        d = []
        for sign in [1, -1]:
            sc = {m: float(np.exp(sign * alpha * v[i])) for i, m in enumerate(MECH)}
            o = run(sc)
            if o:
                d.append(behav_dist(o))
        moves[f'{name}_alpha{alpha}'] = float(np.mean(d)) if d else None
    # single-parameter move (syn alone), matched alpha
    d = []
    for sign in [1, -1]:
        o = run({'syn': float(np.exp(sign * alpha))})
        if o:
            d.append(behav_dist(o))
    moves[f'syn_alone_alpha{alpha}'] = float(np.mean(d)) if d else None

out = {
    'experiment': 'CP_coupling_vs_unused (REAL modWorm)',
    'question': 'is sloppy = unused parameters (A) or coupled/degenerate combinations (B)',
    'hessian_eigvals': [float(x) for x in w],
    'stiff_eigvec_loadings': dict(zip(MECH, [round(float(x), 3) for x in v_stiff])),
    'sloppy_eigvec_loadings': dict(zip(MECH, [round(float(x), 3) for x in v_sloppy])),
    'stiff_eigvec_participation': participation(v_stiff),
    'sloppy_eigvec_participation': participation(v_sloppy),
    'behaviour_change_under_moves': moves,
    'interpretation': {
        'stiff_is_combination': participation(v_stiff) > 1.5,
        'sloppy_combination_preserves_behaviour': None,  # filled below
    },
    'elapsed_sec': round(time.time() - t0, 1),
}
# verdict: sloppy eigvec move should change behaviour far less than stiff eigvec / single param
sm = moves.get('sloppy_eigvec_alpha0.6'); st = moves.get('stiff_eigvec_alpha0.6'); sa = moves.get('syn_alone_alpha0.6')
if sm is not None and st is not None and sm > 0:
    out['interpretation']['sloppy_combination_preserves_behaviour'] = (st / sm > 1.5)
    out['stiff_over_sloppy_ratio_alpha0.6'] = st / sm
    out['single_param_over_sloppy_ratio'] = (sa / sm) if sa else None
json.dump(out, open('/root/autodl-tmp/CP_coupling.json', 'w'), indent=2)
print('=== CP coupling vs unused ===', flush=True)
print('Hessian eigvals:', [round(float(x), 3) for x in w], flush=True)
print('STIFF eigvec loadings:', out['stiff_eigvec_loadings'], '| participation %.2f (1=single,7=all)' % participation(v_stiff), flush=True)
print('SLOPPY eigvec loadings:', out['sloppy_eigvec_loadings'], '| participation %.2f' % participation(v_sloppy), flush=True)
print('behaviour change:', {k: round(v, 4) if v else None for k, v in moves.items()}, flush=True)
print('stiff/sloppy ratio @a0.6:', out.get('stiff_over_sloppy_ratio_alpha0.6'), '| single-param/sloppy:', out.get('single_param_over_sloppy_ratio'), flush=True)
print('CP_DONE %.0fs' % (time.time() - t0), flush=True)
