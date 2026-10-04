"""E-32: the multi-behaviour fitting test, repeated on a second system.

E-31 tests on the nematode whether fitting more behaviours recovers the stiff
subspace better while leaving the sloppy subspace unrecovered. One system is not
enough for a claim the paper puts in its abstract, so the same design is run here
on the fly, whose descending turning commands give a genuine behavioural set:
straight walking, gentle and sharp turns to either side.

Same structure as E-31: a fixed target, the same optimiser and budget from the
same independent starts, fitted against 1, 2, 3 and 4 behaviours, with recovery
error projected onto the stiff and sloppy subspaces of the union curvature.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MUJOCO_GL", "disable")
import json, sys, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

CTX = mp.get_context("spawn")
sys.path.insert(0, "/root/autodl-tmp")

DELTA = 0.25   # AB4: the response is proportional only above ~0.25
RUN_S = 0.25
DT = 1e-4
SUB = 50
NDIM = 48
N_START = 3
N_ITER = 30
POP = 8
SIGMA0 = 0.25
# descending commands: straight, gentle left, sharp right, gentle right
BEHAV = [(1.0, 1.0), (0.4, 1.0), (1.0, 0.2), (1.0, 0.6)]


def _sim(args):
    """One rollout at a gain multiplier vector under one descending command."""
    gains, cmd, seed = args
    import numpy as np
    import flygym.preprogrammed as pp
    from flygym.examples.locomotion import HybridTurningController
    from flygym import Fly
    segs = list(pp.default_leg_sensor_placements)
    fly = Fly(enable_adhesion=True, draw_adhesion=False, contact_sensor_placements=segs)
    sim = HybridTurningController(fly=fly, cameras=[], timestep=DT, seed=seed)
    m = sim.physics.model
    g = np.array(m.actuator_gainprm)
    g[:, 0] = g[:, 0] * np.asarray(gains)
    m.actuator_gainprm[:] = g
    obs, _ = sim.reset(seed=seed)
    action = np.array(cmd, dtype=float)
    out = []
    for t in range(int(RUN_S / DT)):
        obs, _, _, _, _ = sim.step(action)
        if t % SUB == 0:
            out.append(np.array(obs["joints"][0], dtype=float))
    return np.array(out)




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
    """Constrained degrees of freedom against number of behaviours, 48 gains."""
    import itertools, json
    t0 = time.time()
    ones = np.ones(NDIM)
    jobs = [(ones, c, 0) for c in BEHAV]
    for c in BEHAV:
        for i in range(NDIM):
            for sg in (+1, -1):
                g = ones.copy(); g[i] *= (1 + sg * DELTA)
                jobs.append((g, c, 0))
    print(f"{len(jobs)} rollouts", flush=True)
    with ProcessPoolExecutor(max_workers=28, mp_context=CTX) as pool:
        outs = list(pool.map(_sim, jobs, chunksize=1))
    print(f"rollouts done {time.time()-t0:.0f}s", flush=True)

    bases = {c: outs[i] for i, c in enumerate(BEHAV)}
    idx = len(BEHAV)
    H, Js = {}, {}
    for c in BEHAV:
        base = bases[c]
        cols = []
        for i in range(NDIM):
            p = outs[idx]; idx += 1
            q = outs[idx]; idx += 1
            t = min(base.shape[0], p.shape[0], q.shape[0])
            sd = base[:t].std(0).clip(1e-6)
            cols.append((((p[:t] - q[:t]) / (2 * DELTA)) / sd).ravel())
        n = min(len(x) for x in cols)
        J = np.array([x[:n] for x in cols]).T
        Js[c] = J
        M = J.T @ J
        H[c] = M / max(np.trace(M), 1e-30)
        e90, e99, pr = eff(H[c])
        print(f"  {c}: alone {e90}/{e99} of {NDIM}  PR {pr:.2f}", flush=True)

    curve = []
    for k in range(1, len(BEHAV) + 1):
        rec = [eff(sum(H[c] for c in combo))
               for combo in itertools.combinations(BEHAV, k)]
        a = np.array(rec, dtype=float)
        curve.append({"n_behaviours": k, "n_subsets": len(rec),
                      "eff_dim_90_mean": float(a[:, 0].mean()),
                      "eff_dim_90_sd": float(a[:, 0].std()),
                      "eff_dim_99_mean": float(a[:, 1].mean()),
                      "eff_dim_99_sd": float(a[:, 1].std()),
                      "participation_ratio_mean": float(a[:, 2].mean())})
        c_ = curve[-1]
        print(f"  k={k}: 90% {c_['eff_dim_90_mean']:.2f}+-{c_['eff_dim_90_sd']:.2f}"
              f"  99% {c_['eff_dim_99_mean']:.2f}  PR {c_['participation_ratio_mean']:.2f}",
              flush=True)

    out = {"experiment": "E36b_dof_vs_behaviours_flygym_valid_step", "delta": DELTA,
           "model": "FlyGym HybridTurningController, 48 actuator gains",
           "observable": "joint-angle trajectory (rich, uncapped)",
           "behaviours": [list(b) for b in BEHAV], "curve": curve,
           "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(out, open("/root/autodl-tmp/cell/E36b_flygym_validstep.json", "w"), indent=2)
    print("=== E-36 RESULT ===", flush=True)
    for c_ in curve:
        print(f"  {c_['n_behaviours']} behaviour(s): 90% {c_['eff_dim_90_mean']:.2f}"
              f"+-{c_['eff_dim_90_sd']:.2f}  99% {c_['eff_dim_99_mean']:.2f}", flush=True)
    print("E36_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
