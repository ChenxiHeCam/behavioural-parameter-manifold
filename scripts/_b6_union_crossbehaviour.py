'''B6: worm UNION proof. Compute the 20-channel GN Hessian under TWO behaviours --
chemotaxis (food close, interaction_mode online) and plain locomotion (food far, weak gradient) --
and show the UNION (combined Fisher information H_chemo + H_loco) has higher effective dimension
than either single behaviour, i.e. the two behaviours constrain COMPLEMENTARY parameter subspaces.
Run: xvfb-run -a python3 _baai_b6_union.py'''
import os, sys, json, time, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'rms_heading', 'fwd_world']
ION = ['gbshl1','gbshk1','gbkvs1','gbegl2','gbegl36','gbkqt3','gbegl19','gbunc2','gbcca1',
       'gbslo1_egl19','gbslo1_unc2','gbslo2_egl19','gbslo2_unc2','gbkcnl','gbnca','gbirk']
CHANNELS = ION + ['syn', 'gj', 'wout', 'gpas']
DELTA = 0.25; NSTEP = 120
WORKER = '/root/_baai_perturb_worker_orig.py'           # unified worker: food + interaction_mode + channels
FOOD_CLOSE = [1.8275, -0.0276, -0.3082]                 # chemotaxis (strong gradient at worm)
FOOD_FAR   = [1.5, 0.2, -0.5]                            # weak gradient = plain locomotion

def spec_for(channel, factor, food):
    s = {'syn': 1.0, 'gj': 1.0, 'wout': 1.0, 'ion': {}, 'passive': {}, 'n_steps': NSTEP,
         'food_xyz': food, 'interaction_mode': 'online'}
    if channel in ION:      s['ion'] = {channel: factor}
    elif channel == 'gpas': s['passive'] = {'gpas': factor}
    elif channel is not None: s[channel] = factor
    return s

def run(spec):
    p = subprocess.run(['xvfb-run', '-a', 'python3', WORKER, json.dumps(spec)],
                       capture_output=True, text=True, timeout=1800)
    for ln in p.stdout.splitlines():
        if ln.startswith('RESULT'):
            return json.loads(ln[7:])['obs']
    return None

def effdim(eigs, thr):
    a = np.sort(np.abs(np.asarray(eigs, float)))[::-1]; a = a[a > 0]
    return int(np.searchsorted(np.cumsum(a) / a.sum(), thr) + 1) if a.sum() else 0

def hessian_for(food, tag):
    jobs = [(f'{tag}_base', spec_for(None, 1.0, food))]
    for c in CHANNELS:
        jobs.append((f'{tag}_{c}+', spec_for(c, float(np.exp(DELTA)), food)))
        jobs.append((f'{tag}_{c}-', spec_for(c, float(np.exp(-DELTA)), food)))
    res = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for k, o in ex.map(lambda it: (it[0], run(it[1])), jobs):
            res[k] = o; print(f'{k} {time.time()-t0:.0f}s', flush=True)
    base = res[f'{tag}_base']; bv = np.array([base[k] for k in OBS], float); scale = np.abs(bv) + 1e-9
    J = np.zeros((len(OBS), len(CHANNELS)))
    for i, c in enumerate(CHANNELS):
        op, om = res.get(f'{tag}_{c}+'), res.get(f'{tag}_{c}-')
        if op and om:
            J[:, i] = ((np.array([op[k] for k in OBS]) - np.array([om[k] for k in OBS])) / (2 * DELTA)) / scale
    return J, base

t0 = time.time()
J_chemo, base_chemo = hessian_for(FOOD_CLOSE, 'chemo')
J_loco,  base_loco  = hessian_for(FOOD_FAR, 'loco')
H_chemo = J_chemo.T @ J_chemo
H_loco  = J_loco.T  @ J_loco
H_union = H_chemo + H_loco                                # combined Fisher information from both behaviours

w_chemo = np.linalg.eigvalsh(H_chemo); w_loco = np.linalg.eigvalsh(H_loco); w_union = np.linalg.eigvalsh(H_union)
# stiff-subspace overlap between the two behaviours (top-3 eigenvectors)
_, Vc = np.linalg.eigh(H_chemo); _, Vl = np.linalg.eigh(H_loco)
Sc = Vc[:, -3:]; Sl = Vl[:, -3:]
overlap = float(np.linalg.norm(Sc.T @ Sl) / np.sqrt(3))   # 0=orthogonal subspaces, 1=identical

out = {
  'experiment': 'B6_worm_union_proof (BAAIWorm chemotaxis vs locomotion, 20-channel Hessian)',
  'channels': CHANNELS, 'delta': DELTA,
  'eff_dim_chemotaxis_90_99': [effdim(w_chemo, .90), effdim(w_chemo, .99)],
  'eff_dim_locomotion_90_99': [effdim(w_loco, .90), effdim(w_loco, .99)],
  'eff_dim_UNION_90_99':      [effdim(w_union, .90), effdim(w_union, .99)],
  'stiff_subspace_overlap_top3_cos': overlap,
  'eig_chemo': [float(x) for x in np.sort(w_chemo)[::-1][:6]],
  'eig_loco':  [float(x) for x in np.sort(w_loco)[::-1][:6]],
  'eig_union': [float(x) for x in np.sort(w_union)[::-1][:6]],
  'base_net_disp': {'chemotaxis': float(base_chemo['net_disp']), 'locomotion': float(base_loco['net_disp'])},
  'elapsed_s': round(time.time() - t0, 1),
}
json.dump(out, open('/root/paper2_B6_union.json', 'w'), indent=2)
print(f"\nchemotaxis eff {out['eff_dim_chemotaxis_90_99']} | locomotion {out['eff_dim_locomotion_90_99']} | UNION {out['eff_dim_UNION_90_99']} (of 20)")
print(f"stiff-subspace overlap (top3) cos={overlap:.3f}  (low=complementary)")
print('SAVED /root/paper2_B6_union.json', flush=True)
