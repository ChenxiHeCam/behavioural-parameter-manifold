"""modWorm posture worker: one rollout under per-mechanism scalings -> body tangent-angle
matrix (T, L) resampled to common body resolution L. Orientation-invariant (mean angle removed).
Output: RESULT {shape, b64(gzip(float32 angles))}."""
import sys, types, json, os, io, gzip, base64
os.chdir('/root/autodl-tmp/modWorm_fresh')
jm = types.ModuleType('julia'); jm.Main = types.SimpleNamespace(eval=lambda *a, **k: None); jm.api = types.ModuleType('julia.api')
sys.modules['julia'] = jm; sys.modules['julia.api'] = jm.api; sys.path.insert(0, '.')
import numpy as np
from modWorm import network_params as nps, predefined_classes_nv as pnv, predefined_classes_mb as pmb, proprioception_simulation as psim
sp = json.loads(sys.argv[1]); NSTEP = int(sp.get('n_steps', 250)); L = int(sp.get('L', 48))
CE = nps.CE
CE.gap_conductances = CE.gap_conductances * sp.get('gap', 1.0)
CE.syn_conductances = CE.syn_conductances * sp.get('syn', 1.0)
CE.leak_conductances = CE.leak_conductances * sp.get('leak', 1.0)
CE.cell_caps = CE.cell_caps * sp.get('Cm', 1.0)
CE.synaptic_rise_tau = CE.synaptic_rise_tau * sp.get('rise', 1.0)
CE.synaptic_fall_tau = CE.synaptic_fall_tau * sp.get('fall', 1.0)
CE.B = CE.B * sp.get('B', 1.0)
gap = np.load('/root/autodl-tmp/mw_conn_gap_c302.npy'); syn = np.load('/root/autodl-tmp/mw_conn_syn_c302.npy'); mm = np.load('/root/autodl-tmp/mw_muscle_map96.npy')
nv = pnv.CelegansWorm_NervousSystem_PPC(gap, syn); mb = pmb.CelegansWorm_MuscleBody_PPC(mm)
stim = np.load('/root/autodl-tmp/modWorm_fresh/modWorm/presets_input/input_mat_gentle_post_touch.npy')
sol = psim.run_network(nv, mb, stim[:NSTEP])
x = np.asarray(sol['x_solution']); y = np.asarray(sol['y_solution'])
T, P = x.shape
ang = np.zeros((T, L), np.float32)
sn = np.linspace(0, 1, L)
for t in range(T):
    dx = np.diff(x[t]); dy = np.diff(y[t])
    a = np.unwrap(np.arctan2(dy, dx)); a = a - a.mean()
    s = np.linspace(0, 1, len(a)); ang[t] = np.interp(sn, s, a)
b = base64.b64encode(gzip.compress(ang.tobytes())).decode()
print('RESULT ' + json.dumps({'shape': [T, L], 'b64': b}))
