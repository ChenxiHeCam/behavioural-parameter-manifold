"""worm context-dependent identifiability: do DIFFERENT behaviours (chemotaxis vs plain
locomotion) constrain DIFFERENT worm parameters? BAAIWorm has closed-loop chemotaxis (food -> AWA/AWC
chemosensory neurons -> navigation). Perturb the CHEMOSENSORY cells (AWA/AWC) vs MOTOR cells
(VA/VB/DA/DB) under food-close (chemotaxis active) vs food-far (no gradient = plain locomotion).
Prediction: chemosensory-cell perturbation changes behaviour MUCH more WITH food than without (silent
without a gradient); motor-cell perturbation matters in both. -> chemotaxis lights up chemosensory
params that locomotion alone cannot constrain (the worm analogue of the larva multi-module result)."""
import subprocess, json, time
import numpy as np

W = '/root/_baai_perturb_worker_cells.py'
OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'rms_heading', 'fwd_world']
CHEMO = ['AWA', 'AWC']
MOTOR = ['VA', 'VB', 'DA', 'DB']
FOOD_CLOSE = [1.8275, -0.0276, -0.3082]
FOOD_FAR = [50.0, 0.0, 0.0]
PERT = {'passive': {'gpas': 1.6}}   # scale leak conductance of the targeted cells (excitability)
NSTEP = 80


def run(spec):
    spec = dict(spec); spec['n_steps'] = NSTEP
    try:
        p = subprocess.run(['xvfb-run', '-a', 'python3', W, json.dumps(spec)],
                           capture_output=True, text=True, timeout=600)
        for ln in p.stdout.splitlines():
            if ln.startswith('RESULT'):
                return json.loads(ln[7:])['obs']
    except Exception as e:
        print('run err', repr(e)[:80], flush=True)
    return None


def bdist(o, base):
    b = np.array([base[k] for k in OBS]); s = np.abs(b) + 1e-9
    return float(np.sqrt(np.mean(((np.array([o[k] for k in OBS]) - b) / s) ** 2)))


t0 = time.time()
conds = {}
for food, fl in [('close', FOOD_CLOSE), ('far', FOOD_FAR)]:
    base = run({'food_xyz': fl})
    print(f'{food} base done ({time.time()-t0:.0f}s)', flush=True)
    chemo = run({'food_xyz': fl, 'cells': CHEMO, **PERT})
    print(f'{food} chemo done ({time.time()-t0:.0f}s)', flush=True)
    motor = run({'food_xyz': fl, 'cells': MOTOR, **PERT})
    print(f'{food} motor done ({time.time()-t0:.0f}s)', flush=True)
    if base and chemo and motor:
        conds[food] = {'base_net_disp': base['net_disp'],
                       'chemo_perturb_behav_change': bdist(chemo, base),
                       'motor_perturb_behav_change': bdist(motor, base)}
        print(f'{food}: chemo {conds[food]["chemo_perturb_behav_change"]:.4f} | motor {conds[food]["motor_perturb_behav_change"]:.4f}', flush=True)

out = {'experiment': 'BAAIWorm context-dependent identifiability (chemotaxis vs locomotion)',
       'prediction': 'chemosensory(AWA/AWC) perturbation matters WITH food (chemotaxis), silent WITHOUT (locomotion); motor matters in both',
       'conditions': conds}
if 'close' in conds and 'far' in conds:
    cc = conds['close']['chemo_perturb_behav_change']; cf = conds['far']['chemo_perturb_behav_change']
    mc = conds['close']['motor_perturb_behav_change']; mf = conds['far']['motor_perturb_behav_change']
    out['chemo_food_vs_nofood_ratio'] = float(cc / cf) if cf > 1e-9 else None
    out['motor_food_vs_nofood_ratio'] = float(mc / mf) if mf > 1e-9 else None
    out['verdict'] = ('chemosensory params are context-dependent (lit by chemotaxis) if chemo ratio >> motor ratio')
json.dump(out, open('/root/autodl-tmp/BAAI_chemo.json', 'w'), indent=2)
print('=== BAAIWorm context-dependent identifiability ===', flush=True)
print('conditions:', json.dumps(conds, indent=2), flush=True)
print('chemo food/nofood ratio:', out.get('chemo_food_vs_nofood_ratio'), '| motor food/nofood ratio:', out.get('motor_food_vs_nofood_ratio'), flush=True)
print('BAAICHEMO_DONE %.0fs' % (time.time() - t0), flush=True)
