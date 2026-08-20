"""E-37: is the low dimensionality real, or an artefact of coarse-graining?

Every nematode result in the paper is computed over seven global multipliers, one
per mechanism class, applied uniformly to all 279 neurons. An effective dimension
of two or three out of seven is then reported as a statement about the model. But
a seven-dimensional parameterisation cannot have more than seven dimensions of
curvature whatever the model does, and lumping 279 neurons into one multiplier
removes by construction every direction in which neurons differ from each other.

This probes the same model at cell resolution: the leak conductance of each of the
279 neurons is perturbed individually, giving a 279-parameter Jacobian, read out
through the full posture trajectory so the observable count does not cap it either.

If the effective dimension stays near three, the low dimensionality is a property
of the model. If it rises with the parameter count, the published number describes
the parameterisation rather than the nervous system.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json, subprocess, sys, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

CTX = mp.get_context("spawn")
PY = "/root/miniconda3/bin/python"
WORKER = "/root/autodl-tmp/mw_cell_rollout.py"
DELTA = 0.25
NSTEP = 150
BEHAV = "input_mat_gentle_post_touch"


def rollout(a):
    idx, sign = a
    spec = {"cell": int(idx), "sign": int(sign), "delta": DELTA,
            "stim": BEHAV, "n_steps": NSTEP}
    try:
        p = subprocess.run([PY, WORKER, json.dumps(spec)],
                           capture_output=True, text=True, timeout=600)
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                return idx, sign, np.array(json.loads(ln[7:])["phi"], dtype=float)
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
    N = 279
    jobs = [(-1, 0)] + [(i, s) for i in range(N) for s in (+1, -1)]
    print(f"{len(jobs)} rollouts at cell resolution", flush=True)
    store = {}
    with ProcessPoolExecutor(max_workers=40, mp_context=CTX) as pool:
        for n, (idx, sign, phi) in enumerate(pool.map(rollout, jobs, chunksize=1), 1):
            store[(idx, sign)] = phi
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    base = store.get((-1, 0))
    if base is None:
        print("no baseline", flush=True)
        return
    cols, used = [], []
    for i in range(N):
        p, q = store.get((i, +1)), store.get((i, -1))
        if p is None or q is None:
            continue
        t = min(base.shape[0], p.shape[0], q.shape[0])
        sd = base[:t].std(0).clip(1e-6)
        cols.append((((p[:t] - q[:t]) / (2 * DELTA)) / sd).ravel())
        used.append(i)
    n = min(len(c) for c in cols)
    J = np.array([c[:n] for c in cols]).T
    H = J.T @ J
    e90, e99, pr = eff(H)

    # how much curvature does the uniform direction alone carry? that is the
    # single direction the seven-multiplier parameterisation can see
    u = np.ones(len(used)) / np.sqrt(len(used))
    frac_uniform = float(u @ H @ u / max(np.trace(H), 1e-30))

    # per-cell elasticity, for the biological read-out
    elast = np.linalg.norm(J, axis=0)
    order = np.argsort(elast)[::-1]

    out = {"experiment": "E37_parameter_granularity_modworm",
           "question": "does the low effective dimension survive at cell resolution?",
           "n_params": len(used), "parameter": "per-neuron leak conductance",
           "observable": "body tangent-angle trajectory (rich)",
           "delta": DELTA, "behaviour": BEHAV,
           "eff_dim_90": e90, "eff_dim_99": e99, "participation_ratio": pr,
           "fraction_of_curvature_in_uniform_direction": frac_uniform,
           "published_seven_multiplier": {"eff_dim_90": 2, "eff_dim_99": 2,
                                          "participation_ratio": 1.29},
           "top_cells_by_elasticity": [int(used[i]) for i in order[:15]],
           "elasticity_range": [float(elast.min()), float(elast.max())],
           "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(out, open("/root/autodl-tmp/E37_granularity.json", "w"), indent=2)
    print("\n=== E-37 RESULT ===", flush=True)
    print(f"  {len(used)} per-neuron parameters, rich observables", flush=True)
    print(f"  eff-dim {e90} (90%) / {e99} (99%) of {len(used)}   PR {pr:.2f}",
          flush=True)
    print(f"  seven-multiplier version gave 2/2 of 7, PR 1.29", flush=True)
    print(f"  fraction of curvature in the uniform direction: {frac_uniform:.4f}",
          flush=True)
    print("E37_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
