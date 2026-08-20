"""One modWorm rollout, emitting the body tangent-angle trajectory as RESULT JSON."""
import sys, types, json, time, os, warnings

warnings.filterwarnings("ignore")
os.chdir("/root/autodl-tmp/modWorm_fresh")

_j = types.ModuleType("julia")
_j.Main = types.SimpleNamespace(eval=lambda *a, **k: None)
_j.api = types.ModuleType("julia.api")
sys.modules["julia"] = _j
sys.modules["julia.api"] = _j.api
sys.path.insert(0, ".")

import numpy as np
from modWorm import network_params as nps
from modWorm import predefined_classes_nv as pnv, predefined_classes_mb as pmb
from modWorm import proprioception_simulation as psim

spec = json.loads(sys.argv[1])
t0 = time.time()

CE = nps.CE
cell = int(spec.get("cell", -1))
if cell >= 0:
    # perturb one neuron's leak conductance, leaving every other cell untouched
    lk = np.array(CE.leak_conductances, dtype=float)
    lk[cell] = lk[cell] * (1.0 + spec["sign"] * spec["delta"])
    CE.leak_conductances = lk
CE.gap_conductances = CE.gap_conductances * spec.get("gap", 1.0)
CE.syn_conductances = CE.syn_conductances * spec.get("syn", 1.0)
CE.leak_conductances = CE.leak_conductances * spec.get("leak", 1.0) if cell < 0 else CE.leak_conductances
CE.cell_caps = CE.cell_caps * spec.get("Cm", 1.0)
CE.synaptic_rise_tau = CE.synaptic_rise_tau * spec.get("rise", 1.0)
CE.synaptic_fall_tau = CE.synaptic_fall_tau * spec.get("fall", 1.0)
CE.B = CE.B * spec.get("B", 1.0)

gap = np.load("modWorm/data/conn_gap_adjust_Cook.npy")
syn = np.load("modWorm/data/conn_syn_adjust_Cook.npy")
mmap = np.load("modWorm/muscle_maps/muscle_map_adjust.npy")

nv = pnv.CelegansWorm_NervousSystem_PPC(gap, syn)
mb = pmb.CelegansWorm_MuscleBody_PPC(mmap)

stim = np.load(f"modWorm/presets_input/{spec.get('stim','input_mat_gentle_post_touch')}.npy")
nstep = int(spec.get("n_steps", 150))
inp = stim[:nstep] if stim.ndim > 1 else np.tile(stim, (nstep, 1))

out = psim.run_network(nv, mb, inp)
phi = np.asarray(out["phi"], dtype=float)
sub = max(1, phi.shape[0] // 120)
print("RESULT " + json.dumps({"phi": phi[::sub].tolist(),
                              "sec": round(time.time() - t0, 2)}), flush=True)
