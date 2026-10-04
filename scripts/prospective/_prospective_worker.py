"""One BAAIWorm point-circuit rollout with explicit output coordinates.

JSON argv1 specifies connection or named mechanism gain, simulation duration,
and observable. motor_voltage is the default. muscle_preactivation is V @ W;
muscle_activation is the native clip((V @ W + 80)/100, 0, .8).
The ambiguous label muscle is rejected. Named gains use the mechanism SUFFIX
and RANGE variable explicitly; absence is a simulation error.
RESULT keeps the motor array key for compatibility and records observable.
"""
import sys, json, time, os, pickle, warnings

warnings.filterwarnings("ignore")
ROOT = os.environ.get("BAAI_ROOT", "/root/autodl-tmp/BAAIWorm_fresh")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eworm", "ghost_in_mesh_sim"))
os.chdir(ROOT)

import numpy as np
from eworm.network import transform
from eworm.utils import data_factory

spec = json.loads(sys.argv[1])
t0 = time.time()

cfg = json.load(open("eworm/network/config.json"))
if os.environ.get("BAAI_MECHANISM_DIR"):
    cfg["dir_info"]["mechanism_dir"] = os.environ["BAAI_MECHANISM_DIR"]
GRP = spec.get("group", "video_offline")
GD = f"eworm/ghost_in_mesh_sim/data/tuned/{GRP}"
abs_c = pickle.load(open(f"{GD}/{GRP}_abscircuit.pkl", "rb"))

# displace the whole weight vector to a new operating point first, so the
# curvature can be evaluated away from the published parameters
pf = float(spec.get("point_fold", 0) or 0)
if pf > 1.0:
    rng = np.random.default_rng(int(spec.get("point_seed", 0)))
    lf = np.log(pf)
    for c_ in abs_c.connections:
        c_.weight = c_.weight * float(np.exp(rng.uniform(-lf, lf)))

conn = int(spec.get("conn", -1))
if conn >= 0:
    c = abs_c.connections[conn]
    c.weight = c.weight * (1.0 + spec["sign"] * spec["delta"])

circuit = transform.abstract2point(abs_c, cfg)

# a named mechanism is scaled across every cell that expresses it, which is the
# granularity at which the biological reading is made. The conductance is set on
# the NEURON sections directly, after construction, so no parameter file is touched.
chan = spec.get("channel")
if chan:
    from neuron import h as _h
    factor = 1.0 + spec["sign"] * spec["delta"]
    from _baai_worker_support import scale_conductance
    touched = scale_conductance(_h.allsec(), chan, spec.get("mech"), factor)

# Prospective coordinates: explicit log gains, jointly applied at an operating point.
from _baai_worker_support import scale_conductance
from neuron import h as _h
gain_touched={}
for mechanism,coordinate in spec.get("log_gains", {}).items():
    gain_touched[mechanism]=scale_conductance(_h.allsec(), "gb"+mechanism, mechanism, float(np.exp(coordinate)))

tstop = float(spec.get("tstop", 3000))
sim_config = {"dt": 5 / 3, "tstop": tstop, "v_init": -65, "secondorder": 0}
# the sensory and motor cell lists the model itself uses, from worm_in_env.py
in_names = ["AWAL", "AWAR", "AWCL", "AWCR", "ASKL", "ASKR", "ALNL", "ALNR",
            "PLML", "PHAL", "PHAR", "URYDL", "URYDR", "URYVL", "URYVR"]
out_names = ["RIML", "RIMR", "RMEL", "RMER", "RMED", "RMEV", "RMDDL", "RMDDR",
             "RMDL", "RMDR", "RMDVL", "RMDVR", "RIVL", "RIVR", "SMDDL", "SMDDR",
             "SMDVL", "SMDVR", "SMBDL", "SMBDR", "SMBVL", "SMBVR"] +             [f"DA{i:02d}" for i in range(1, 10)] +             [f"DB{i:02d}" for i in range(1, 8)] +             [f"DD{i:02d}" for i in range(1, 7)] +             [f"VA{i:02d}" for i in range(1, 13)] +             [f"VB{i:02d}" for i in range(1, 12)] +             [f"VD{i:02d}" for i in range(1, 14)]

# a behaviour is defined by which sensory modality is driven: different modalities
# activate different sensory neurons and so are expected to constrain different
# parts of the circuit
MODALITY = {
    "olfactory":      ["AWAL", "AWAR", "AWCL", "AWCR"],
    "chemosensory":   ["ASKL", "ASKR", "PHAL", "PHAR"],
    "touch_anterior": ["ALNL", "ALNR"],
    "touch_posterior": ["PLML"],
    "proprioceptive": ["URYDL", "URYDR", "URYVL", "URYVR"],
}

loc_path = f"{GD}/{GRP}_loc.npy"
if os.path.exists(loc_path):
    inp = data_factory.ghost_in_mesh_data_factory(
        len(in_names), sim_config["tstop"], sim_config["dt"], loc_path)
else:
    n = int(tstop / sim_config["dt"])
    t = np.arange(n) * sim_config["dt"] / 1000.0
    inp = np.stack([2.0 * np.sin(2 * np.pi * (0.5 * t + i / len(in_names)))
                    for i in range(len(in_names))])

# a behaviour is a drive pattern, not only a choice of neurons: a touch is a brief
# transient on the mechanosensory cells, foraging a slow gradient on the
# chemosensory cells, and the two differ in time course as well as in target
PATTERN = {
    "touch_anterior_tap": (["ALNL", "ALNR"], "pulse"),
    "touch_posterior_tap": (["PLML"], "pulse"),
    "chemical_foraging": (["AWAL", "AWAR", "AWCL", "AWCR"], "gradient"),
    "avoidance_chemical": (["ASKL", "ASKR", "PHAL", "PHAR"], "gradient"),
    "proprioceptive_drive": (["URYDL", "URYDR", "URYVL", "URYVR"], "oscillation"),
}
pat = spec.get("pattern")
if pat:
    cells, kind = PATTERN[pat]
    n = inp.shape[1]
    tt = np.arange(n) * sim_config["dt"] / 1000.0
    if kind == "pulse":
        w = np.exp(-((tt - 0.25) ** 2) / (2 * 0.03 ** 2)) * 6.0
    elif kind == "gradient":
        w = 2.0 * (1.0 - np.exp(-tt / 0.6))
    else:
        w = 2.0 * np.sin(2 * np.pi * 2.0 * tt)
    inp = np.zeros_like(inp)
    for i, nm in enumerate(in_names):
        if nm in cells:
            inp[i] = w

# Explicit voltage delivered to the external synapse, whose Vth is 0 mV.
# These are model interventions, not calibrated animal sensory stimuli.
stim=spec.get("stimulus")
if stim:
    n=int(tstop/sim_config["dt"])
    tt=np.arange(n)*sim_config["dt"]
    off=float(stim["off_mV"])
    inp=np.full((len(in_names),n),off,dtype=float)
    for component in stim["components"]:
        begin=float(component["start_ms"]);end=float(component["end_ms"])
        active=(tt>=begin)&(tt<end)
        wave=np.full(n,off,dtype=float)
        if component["shape"]=="step":
            wave[active]=float(component["on_mV"])
        elif component["shape"]=="ramp":
            wave[active]=off+(float(component["on_mV"])-off)*(tt[active]-begin)/(end-begin)
        else:raise ValueError("Unsupported stimulus shape")
        for cell in component["cells"]:
            if cell not in in_names:raise ValueError("Unknown stimulated cell "+cell)
            inp[in_names.index(cell)]=wave

# the input path has to be wired explicitly: the simulation injects through
# Connection objects on the circuit, not by matching cell names
from eworm.network import detailed_circuit as dc
circuit.input_connections = []
for cn in in_names:
    circuit.add_connection(
        dc.Connection(None, circuit.cell(cell_name=cn).segments[0], "syn", 10))

mods = spec.get("modalities")
if mods:
    driven = set()
    for m in mods:
        driven.update(MODALITY[m])
    mask = np.array([1.0 if n in driven else 0.0 for n in in_names])[:, None]
    inp = np.asarray(inp, dtype=float) * mask

out = circuit.simulation(sim_config, inp, in_names, out_names)
out = np.asarray(out, dtype=float)

# Output coordinates are explicit. Preactivation and clipped activation are
# different observables and must never share a result label.
from _baai_worker_support import select_readout
V = out if out.shape[0] == len(out_names) else out.T
observable = spec.get("observable", "motor_voltage")
wout = None
if observable != "motor_voltage":
    with open(f"{GD}/{GRP}_wout.pkl", "rb") as f:
        wout = pickle.load(f)
readout = select_readout(V, observable, wout)
sub = max(1, readout.shape[1] // 120)
print("RESULT " + json.dumps({"motor": readout[:, ::sub].tolist(),
                              "observable": observable,
                              "n_cells": int(readout.shape[0]),
                              "n_conductance_segments_touched": touched if chan else 0,
                              "gain_segments_touched": gain_touched, "stimulus": stim, "sec": round(time.time() - t0, 2)}), flush=True)
