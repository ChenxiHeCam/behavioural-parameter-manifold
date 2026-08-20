"""One BAAIWorm circuit rollout under a scaling of one connection weight.

The 136-cell, 3076-connection circuit is built from the tuned abstract circuit
shipped with the repository and run as point neurons in NEURON. The observable is
the motor output: soma voltages of the output cells mapped through the model's own
readout to the 96 muscle activations, which is the last quantity before body
mechanics and so the closest thing to behaviour that can be computed without the
compiled physics engine.

argv1: {"conn": index or -1, "sign": +1/-1, "delta": float, "tstop": ms}
Prints RESULT {"muscle": [[...]], "sec": float}
"""
import sys, json, time, os, pickle, warnings

warnings.filterwarnings("ignore")
ROOT = "/root/autodl-tmp/BAAIWorm_fresh"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eworm", "ghost_in_mesh_sim"))
os.chdir(ROOT)

import numpy as np
from eworm.network import transform
from eworm.utils import data_factory

spec = json.loads(sys.argv[1])
t0 = time.time()

cfg = json.load(open("eworm/network/config.json"))
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
    mech = chan[2:] if chan.startswith("gb") else chan       # gbshl1 -> shl1
    touched = 0
    for sec in _h.allsec():
        try:
            if not sec.has_membrane(mech):
                continue
            for seg in sec:
                m = getattr(seg, mech)
                setattr(m, chan, getattr(m, chan) * factor)
                touched += 1
        except Exception:
            continue
    if touched == 0:
        print(f"WARN channel {chan} not found", file=sys.stderr)

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

# the motor-neuron membrane potentials are the model's motor command, one step
# before the muscle readout; the readout itself passes through a tanh that is
# saturated at this voltage scale and so carries no usable derivative
V = out if out.shape[0] == len(out_names) else out.T
sub = max(1, V.shape[1] // 120)
print("RESULT " + json.dumps({"motor": V[:, ::sub].tolist(),
                              "n_cells": int(V.shape[0]),
                              "sec": round(time.time() - t0, 2)}), flush=True)
