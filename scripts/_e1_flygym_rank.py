"""E-1: is the FlyGym low effective dimension real, or a rank cap from 10 observables?

rank(J^T W J) <= n_obs. The published result probes 48 actuator gains with 10 coarse
scalars and reports effective dimension 3 (90%) / 6 (99%) of 48 -- but with 10
observables the Hessian can have at most 10 non-zero eigenvalues, so "42 flat
directions" is forced by construction and says nothing about the parameter space.

This script measures the same 48 gains twice from the same rollouts:
  coarse : the 10 published scalars          -> reproduces the published number
  rich   : the joint-angle trajectory, T x 42 -> thousands of observables, no cap

If the rich effective dimension stays low, the low-dimensionality is a property of
the model. If it grows toward 48, the published number was an observable artefact.
"""
import json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("MUJOCO_GL", "osmesa")

DELTA = 0.05          # published perturbation size
NSEED = 5             # published seed count
RUN_S = 0.5           # rollout duration (s)
DT = 1e-4
SUB = 50              # subsample stride for the rich observables
import flygym.preprogrammed as _pp
SEGS = list(_pp.default_leg_sensor_placements)   # the 36 segments adhesion expects


def rollout(args):
    """One rollout at a given gain vector; returns coarse scalars and posture matrix."""
    idx, sign, seed = args
    import numpy as np
    from flygym.examples.locomotion import HybridTurningController
    from flygym import Fly

    fly = Fly(enable_adhesion=True, draw_adhesion=False, contact_sensor_placements=SEGS)
    sim = HybridTurningController(fly=fly, cameras=[], timestep=DT, seed=seed)
    m = sim.physics.model
    gain = np.array(m.actuator_gainprm)
    if idx >= 0:
        gain[idx, 0] *= (1.0 + sign * DELTA)
        m.actuator_gainprm[:] = gain

    obs, _ = sim.reset(seed=seed)
    nstep = int(RUN_S / DT)
    pos, joints = [], []
    action = np.array([1.0, 1.0])
    for t in range(nstep):
        obs, _, _, _, _ = sim.step(action)
        if t % SUB == 0:
            pos.append(np.array(obs["fly"][0], dtype=float))       # x,y,z
            joints.append(np.array(obs["joints"][0], dtype=float))  # 42 angles
    pos = np.array(pos)
    joints = np.array(joints)

    # --- coarse: the ten published scalars -------------------------------
    d = np.diff(pos, axis=0) / (SUB * DT)
    sp = np.linalg.norm(d[:, :2], axis=1)
    ee = joints
    coarse = np.array([
        d[:, 0].mean(), d[:, 1].mean(), d[:, 2].mean(),
        sp.mean(), sp.std(),
        np.gradient(np.unwrap(np.arctan2(d[:, 1], d[:, 0]))).mean(),
        np.linalg.norm(pos, axis=1).mean(), np.linalg.norm(pos, axis=1).std(),
        np.corrcoef(ee[:, 0], ee[:, 21])[0, 1] if ee.shape[0] > 2 else 0.0,
        float(((sp - sp.mean()) ** 3).mean() / (sp.std() ** 3 + 1e-12)),
    ], dtype=float)
    coarse = np.nan_to_num(coarse)
    return idx, sign, seed, coarse, joints.astype(np.float32)


def eff_dim(ev):
    ev = np.maximum(np.real(ev), 0.0)
    ev = ev[ev > 0]
    if ev.size == 0:
        return 0, 0, 0.0, []
    s = np.sort(ev)[::-1]
    cs = np.cumsum(s) / s.sum()
    pr = float(s.sum() ** 2 / (s ** 2).sum())
    return (int(np.searchsorted(cs, 0.90) + 1), int(np.searchsorted(cs, 0.99) + 1),
            pr, [float(x) for x in s[:20]])


def main():
    t0 = time.time()
    jobs = [(-1, 0, s) for s in range(NSEED)]
    for i in range(48):
        for sg in (+1, -1):
            for s in range(NSEED):
                jobs.append((i, sg, s))
    print(f"{len(jobs)} rollouts on {os.cpu_count()} cores", flush=True)

    res = {}
    with ProcessPoolExecutor(max_workers=min(48, os.cpu_count() - 4)) as ex:
        for n, out in enumerate(ex.map(rollout, jobs, chunksize=1), 1):
            idx, sign, seed, coarse, joints = out
            res[(idx, sign, seed)] = (coarse, joints)
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    T = min(v[1].shape[0] for v in res.values())
    base_c = np.mean([res[(-1, 0, s)][0] for s in range(NSEED)], axis=0)
    base_j = np.mean([res[(-1, 0, s)][1][:T] for s in range(NSEED)], axis=0)

    Jc, Jr = [], []
    for i in range(48):
        cp = np.mean([res[(i, +1, s)][0] for s in range(NSEED)], axis=0)
        cm = np.mean([res[(i, -1, s)][0] for s in range(NSEED)], axis=0)
        Jc.append((cp - cm) / (2 * DELTA))
        jp = np.mean([res[(i, +1, s)][1][:T] for s in range(NSEED)], axis=0)
        jm = np.mean([res[(i, -1, s)][1][:T] for s in range(NSEED)], axis=0)
        Jr.append(((jp - jm) / (2 * DELTA)).flatten())
    Jc = np.array(Jc).T
    Jr = np.array(Jr).T

    wc = 1.0 / np.maximum(np.abs(base_c), 1e-8)
    Hc = (Jc * wc[:, None]).T @ (Jc * wc[:, None])
    sj = base_j.std(0).clip(1e-6)
    Hr = (Jr / np.tile(sj, T)[:, None]).T @ (Jr / np.tile(sj, T)[:, None])

    c90, c99, cpr, cspec = eff_dim(np.linalg.eigvalsh(Hc))
    r90, r99, rpr, rspec = eff_dim(np.linalg.eigvalsh(Hr))

    out = {
        "experiment": "E1_flygym_rank_control",
        "question": "is the FlyGym effective dimension a property of the model or a "
                    "rank cap imposed by using 10 observables?",
        "n_params": 48, "delta": DELTA, "n_seeds": NSEED,
        "rollout_seconds": RUN_S, "frames_kept": int(T),
        "coarse": {"n_observables": 10, "eff_dim_90": c90, "eff_dim_99": c99,
                   "participation_ratio": cpr, "top_eigenvalues": cspec},
        "rich": {"n_observables": int(T * 42), "eff_dim_90": r90, "eff_dim_99": r99,
                 "participation_ratio": rpr, "top_eigenvalues": rspec},
        "published_coarse_eff_dim_90_99": [3, 6],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open("/root/autodl-tmp/E1_flygym_rank.json", "w"), indent=2)
    print("\n=== E-1 RESULT ===", flush=True)
    print(f"coarse  10 obs      -> eff-dim {c90}/{c99} of 48   PR {cpr:.2f}", flush=True)
    print(f"rich    {T*42} obs  -> eff-dim {r90}/{r99} of 48   PR {rpr:.2f}", flush=True)
    print(f"published coarse    -> eff-dim 3/6 of 48", flush=True)
    print("E1_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
