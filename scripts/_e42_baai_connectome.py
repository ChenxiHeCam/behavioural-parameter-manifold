"""E-42: how many degrees of freedom does the motor output constrain in a
connectome-scale biophysical model?

BAAIWorm is the current-generation nematode model: 136 multi-compartment cells
reduced to point neurons, 3076 connectome-derived synaptic and gap-junction
weights, published in Nature Computational Science. It is the largest parameter
space in this study by two orders of magnitude, and unlike the older nematode
simulator its response to a weight perturbation is smooth and close to
proportional, so a finite-difference Jacobian is well posed.

Every one of the 3076 connection weights is perturbed individually and the motor
output -- the membrane potentials of the 80 motor neurons over the whole
simulation -- is recorded, giving a Jacobian with thousands of observables and
thousands of parameters. Reported: the effective dimension, the participation
ratio, and how the curvature divides between chemical synapses and gap junctions.

A step-size scan accompanies it, so the working point is shown to lie in a regime
where the response is above the numerical floor and still linear.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json, pickle, subprocess, sys, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

CTX = mp.get_context("spawn")
PY = "/root/miniconda3/bin/python"
WORKER = "/root/autodl-tmp/baai_worker.py"
ROOT = "/root/autodl-tmp/BAAIWorm_fresh"
DELTA = 0.5
TSTOP = 2000
N_CONN = 3076


def rollout(a):
    idx, sign = a
    spec = {"conn": int(idx), "sign": int(sign), "delta": DELTA, "tstop": TSTOP}
    try:
        p = subprocess.run([PY, WORKER, json.dumps(spec)],
                           capture_output=True, text=True, timeout=900)
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                return idx, sign, np.array(json.loads(ln[7:])["motor"], dtype=np.float32)
    except Exception:
        pass
    return idx, sign, None


def eff(H):
    ev = np.sort(np.linalg.eigvalsh(H))[::-1]
    ev = np.maximum(ev, 0)
    if ev.sum() <= 0:
        return 0, 0, 0.0
    cs = np.cumsum(ev) / ev.sum()
    return (int(np.searchsorted(cs, 0.90) + 1),
            int(np.searchsorted(cs, 0.99) + 1),
            float(ev.sum() ** 2 / (ev ** 2).sum()))


def main():
    t0 = time.time()
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "eworm", "ghost_in_mesh_sim"))
    GD = os.path.join(ROOT, "eworm/ghost_in_mesh_sim/data/tuned/video_offline")
    abs_c = pickle.load(open(f"{GD}/video_offline_abscircuit.pkl", "rb"))
    cats = [getattr(c, "category", "?") for c in abs_c.connections]
    print(f"{len(cats)} connections, categories "
          f"{ {c: cats.count(c) for c in set(cats)} }", flush=True)

    jobs = [(-1, 0)] + [(i, s) for i in range(N_CONN) for s in (+1, -1)]
    print(f"{len(jobs)} rollouts", flush=True)
    store = {}
    with ProcessPoolExecutor(max_workers=40, mp_context=CTX) as pool:
        for n, (idx, sign, v) in enumerate(pool.map(rollout, jobs, chunksize=1), 1):
            store[(idx, sign)] = v
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    base = store.get((-1, 0))
    if base is None:
        print("no baseline", flush=True)
        return
    sd = base.std(1).clip(1e-3)[:, None]
    cols, used = [], []
    for i in range(N_CONN):
        p, q = store.get((i, +1)), store.get((i, -1))
        if p is None or q is None:
            continue
        t = min(base.shape[1], p.shape[1], q.shape[1])
        cols.append((((p[:, :t] - q[:, :t]) / (2 * DELTA)) / sd).ravel())
        used.append(i)
    n = min(len(c) for c in cols)
    J = np.array([c[:n] for c in cols], dtype=np.float64).T
    print(f"Jacobian {J.shape}  {time.time()-t0:.0f}s", flush=True)

    H = J.T @ J
    e90, e99, pr = eff(H)
    elast = np.linalg.norm(J, axis=0)
    dead = int((elast < 1e-12 * max(elast.max(), 1e-30)).sum())

    by_cat = {}
    for c in set(cats):
        m = np.array([cats[i] == c for i in used])
        if m.any():
            by_cat[c] = {"n": int(m.sum()),
                         "median_elasticity": float(np.median(elast[m])),
                         "share_of_curvature": float((elast[m] ** 2).sum()
                                                     / max((elast ** 2).sum(), 1e-30))}

    out = {"experiment": "E42_baaiworm_connectome",
           "model": "BAAIWorm point circuit, 136 cells, connectome weights",
           "observable": "motor-neuron membrane potentials, 80 cells x time",
           "n_parameters": len(used), "n_observables": int(J.shape[0]),
           "delta": DELTA, "tstop_ms": TSTOP,
           "eff_dim_90": e90, "eff_dim_99": e99, "participation_ratio": pr,
           "n_parameters_with_no_measurable_effect": dead,
           "elasticity_percentiles": {str(p): float(np.percentile(elast, p))
                                      for p in (1, 25, 50, 75, 99)},
           "by_connection_category": by_cat,
           "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(out, open("/root/autodl-tmp/E42_baai_connectome.json", "w"), indent=2)
    print("=== E-42 RESULT ===", flush=True)
    print(f"  {len(used)} connection weights, {J.shape[0]} observables", flush=True)
    print(f"  eff-dim {e90} (90%) / {e99} (99%)   PR {pr:.2f}", flush=True)
    print(f"  parameters with no measurable effect: {dead}", flush=True)
    for c, v in by_cat.items():
        print(f"  {c}: n={v['n']}  median elasticity {v['median_elasticity']:.3e}"
              f"  share of curvature {v['share_of_curvature']:.3f}", flush=True)
    print("E42_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
