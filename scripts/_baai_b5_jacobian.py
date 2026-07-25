'''B5: BAAIWorm 20-channel observable Jacobian -> GN Hessian eigenVECTORS (not just eigenvalues).
Central finite differences in log-parameter space around the optimum (all multipliers = 1.0).
Saves eigenvalues + eigenvectors (20x20) + channel labels so the stiff/sloppy SUBSPACE can be
projected against real biological covariance (CeNGEN) locally.
Run: xvfb-run -a python3 _baai_b5_jacobian.py'''
import os, sys, json, time, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'rms_heading', 'fwd_world']
ION = ['gbshl1','gbshk1','gbkvs1','gbegl2','gbegl36','gbkqt3','gbegl19','gbunc2','gbcca1',
       'gbslo1_egl19','gbslo1_unc2','gbslo2_egl19','gbslo2_unc2','gbkcnl','gbnca','gbirk']
# 20 named channels: 16 ion + synaptic + gap-junction + motor-output + passive-leak
CHANNELS = ION + ['syn', 'gj', 'wout', 'gpas']
DELTA = 0.25      # log-space step (matches the elasticity probe)
NSTEP = 120

def spec_for(channel, factor):
    s = {'syn': 1.0, 'gj': 1.0, 'wout': 1.0, 'ion': {}, 'passive': {}, 'n_steps': NSTEP}
    if channel in ION:      s['ion'] = {channel: factor}
    elif channel == 'gpas': s['passive'] = {'gpas': factor}
    else:                   s[channel] = factor   # syn / gj / wout
    return s

def run(spec):
    p = subprocess.run(['xvfb-run', '-a', 'python3', '/root/_baai_perturb_worker.py', json.dumps(spec)],
                       capture_output=True, text=True, timeout=1800)
    for ln in p.stdout.splitlines():
        if ln.startswith('RESULT'):
            return json.loads(ln[7:])['obs']
    return None

# build job list: base + (+delta, -delta) per channel
jobs = [('base', spec_for('syn', 1.0))]
for c in CHANNELS:
    jobs.append((f'{c}+', spec_for(c, float(np.exp(DELTA)))))
    jobs.append((f'{c}-', spec_for(c, float(np.exp(-DELTA)))))

t0 = time.time(); res = {}
def do(it):
    tag, sp = it; return tag, run(sp)
with ThreadPoolExecutor(max_workers=10) as ex:
    for tag, o in ex.map(do, jobs):
        res[tag] = o
        print(f'{tag} done {time.time()-t0:.0f}s', flush=True)

base = res['base']
bv = np.array([base[k] for k in OBS], float)
scale = np.abs(bv) + 1e-9                       # observable normalisation (W = diag(1/scale^2))

# central-difference Jacobian J: d(obs_norm)/d(log theta), shape (6 obs, 20 params)
J = np.zeros((len(OBS), len(CHANNELS)))
missing = []
for i, c in enumerate(CHANNELS):
    op, om = res.get(f'{c}+'), res.get(f'{c}-')
    if op is None or om is None:
        missing.append(c); continue
    vp = np.array([op[k] for k in OBS], float)
    vm = np.array([om[k] for k in OBS], float)
    J[:, i] = ((vp - vm) / (2 * DELTA)) / scale

H = J.T @ J                                     # Gauss-Newton Hessian, 20x20
w, V = np.linalg.eigh(H)                        # ascending
order = np.argsort(w)[::-1]
w = w[order]; V = V[:, order]                   # descending: V[:,0] = stiffest eigenvector

aw = np.abs(w); cum = np.cumsum(aw) / aw.sum()
eff90 = int(np.searchsorted(cum, 0.90) + 1)
eff99 = int(np.searchsorted(cum, 0.99) + 1)

out = {
    'experiment': 'B5_baai_perchannel_hessian_eigenVECTORS',
    'channels': CHANNELS, 'observables': OBS, 'delta': DELTA, 'n_steps': NSTEP,
    'base_obs': {k: float(base[k]) for k in OBS},
    'eigenvalues': [float(x) for x in w],
    'eigenvectors': [[float(x) for x in V[:, j]] for j in range(V.shape[1])],  # eigenvectors[j] = j-th, over channels
    'eff_dim_90': eff90, 'eff_dim_99': eff99,
    'jacobian': [[float(x) for x in J[r]] for r in range(J.shape[0])],
    'missing_channels': missing,
    'elapsed_s': round(time.time() - t0, 1),
}
json.dump(out, open('/root/paper2_B5_baai_hessian_eigvec.json', 'w'), indent=2)
# also a compact stiff/sloppy summary
stiff = [CHANNELS[k] for k in np.argsort(np.abs(V[:, 0]))[::-1][:5]]
sloppy_vec = V[:, -1]
print(f'eff_dim {eff90}/{eff99} of 20 | top eigvals {[round(float(x),3) for x in w[:5]]} | missing {missing}', flush=True)
print(f'stiffest-eigvec top channels: {stiff}', flush=True)
print('SAVED /root/paper2_B5_baai_hessian_eigvec.json', flush=True)
