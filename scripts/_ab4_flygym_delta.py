"""AB4: re-measure the FlyGym curvature at steps where the response is proportional.

S1.5 reports that this controller has a stochastic floor of 0.037 in the whitened
joint-angle response, present already at a step of 0.001, and that proportionality
is recovered only above a step of about 0.25. Both deposited probes, however, use
delta = 0.05, which is inside the region S1.5 declares invalid. The reported 16/38
therefore does not pass the paper's own second gate, the same gate that has just
retired the pyloric dimension.

This repeats the rank-control probe of E1 at four steps spanning the stated
boundary, 0.05, 0.15, 0.25 and 0.50, and reports for each: the whitened response
norm and its ratio to the step, which is the proportionality test itself, and the
coarse and rich effective dimensions. If the dimension is stable across the valid
region the published figure survives with a corrected step; if it moves, the number
must be restated at a valid step.

The controller is stochastic, so every leg is averaged over seeds, and the
seed-to-seed null of E1_flygym_null is the floor these responses must clear.
"""
import json, os, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("MUJOCO_GL", "disable")

DELTAS = [0.05, 0.15, 0.25, 0.50]
NSEED = int(os.environ.get("NSEED", "5"))
RUN_S = 0.5
DT = 1e-4
SUB = 50
NW = int(os.environ.get("NW", "28"))
OUT = "/root/autodl-tmp/cell/AB4_flygym_delta.json"

import flygym.preprogrammed as _pp
SEGS = list(_pp.default_leg_sensor_placements)


def rollout(args):
    """One rollout at a given gain vector; returns the ten scalars and the posture."""
    idx, sign, delta, seed = args
    import numpy as np
    from flygym.examples.locomotion import HybridTurningController
    from flygym import Fly

    fly = Fly(enable_adhesion=True, draw_adhesion=False, contact_sensor_placements=SEGS)
    sim = HybridTurningController(fly=fly, cameras=[], timestep=DT, seed=seed)
    m = sim.physics.model
    if idx >= 0:
        gain = np.array(m.actuator_gainprm)
        gain[idx, 0] *= (1.0 + sign * delta)
        m.actuator_gainprm[:] = gain

    # a large gain perturbation can drive the physics divergent, which is itself
    # part of the answer: a step big enough to give a proportional response is not
    # always a step the simulator survives. Such legs are dropped and counted.
    try:
        obs, _ = sim.reset(seed=seed)
        pos, joints = [], []
        action = np.array([1.0, 1.0])
        for t in range(int(RUN_S / DT)):
            obs, _, _, _, _ = sim.step(action)
            if t % SUB == 0:
                pos.append(np.array(obs["fly"][0], dtype=float))
                joints.append(np.array(obs["joints"][0], dtype=float))
    except Exception as e:
        return (idx, sign, delta, seed), ("DIVERGED", type(e).__name__)
    pos, joints = np.array(pos), np.array(joints)

    d = np.diff(pos, axis=0) / (SUB * DT)
    sp = np.linalg.norm(d[:, :2], axis=1)
    ee = joints
    coarse = np.nan_to_num(np.array([
        d[:, 0].mean(), d[:, 1].mean(), d[:, 2].mean(),
        sp.mean(), sp.std(),
        np.gradient(np.unwrap(np.arctan2(d[:, 1], d[:, 0]))).mean(),
        np.linalg.norm(pos, axis=1).mean(), np.linalg.norm(pos, axis=1).std(),
        np.corrcoef(ee[:, 0], ee[:, 21])[0, 1] if ee.shape[0] > 2 else 0.0,
        float(((sp - sp.mean()) ** 3).mean() / (sp.std() ** 3 + 1e-12)),
    ], dtype=float))
    return (idx, sign, delta, seed), (coarse, joints.astype(np.float32))


def eff_dim(H):
    ev = np.maximum(np.real(np.linalg.eigvalsh(H)), 0.0)
    ev = ev[ev > 0]
    if ev.size == 0:
        return 0, 0, 0.0
    s = np.sort(ev)[::-1]
    cs = np.cumsum(s) / s.sum()
    return (int(np.searchsorted(cs, 0.90) + 1), int(np.searchsorted(cs, 0.99) + 1),
            float(s.sum() ** 2 / (s ** 2).sum()))


def main():
    t0 = time.time()
    jobs = [(-1, 0, 0.0, s) for s in range(NSEED)]
    for d in DELTAS:
        for i in range(48):
            for sg in (+1, -1):
                for s in range(NSEED):
                    jobs.append((i, sg, d, s))
    print(f"{len(jobs)} rollouts, {len(DELTAS)} steps x 48 gains x 2 signs x {NSEED} seeds",
          flush=True)

    res = {}
    with ProcessPoolExecutor(max_workers=NW) as ex:
        for n, (k, v) in enumerate(ex.map(rollout, jobs, chunksize=1), 1):
            res[k] = v
            if n % 200 == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    def _bad(v):
        return isinstance(v[0], str)          # "DIVERGED" marker, not an array
    ok = {k: v for k, v in res.items() if not _bad(v)}
    diverged = {k: v[1] for k, v in res.items() if _bad(v)}
    import pickle
    with open(OUT.replace(".json", "_raw.pkl"), "wb") as fh:
        pickle.dump({"ok": {k: (v[0], v[1]) for k, v in ok.items()},
                     "diverged": diverged}, fh, protocol=4)
    if any(k[0] == -1 for k in diverged):
        raise SystemExit("a baseline rollout diverged; refusing to report")
    print(f"{len(diverged)} of {len(res)} legs diverged", flush=True)

    T = min(v[1].shape[0] for v in ok.values())
    base_c = np.mean([res[(-1, 0, 0.0, s)][0] for s in range(NSEED)], axis=0)
    base_j = np.mean([res[(-1, 0, 0.0, s)][1][:T] for s in range(NSEED)], axis=0)
    sj = base_j.std(0).clip(1e-6)
    wc = 1.0 / np.maximum(np.abs(base_c), 1e-8)

    rows = []
    for d in DELTAS:
        Jc, Jr, norms, dropped = [], [], [], 0
        for i in range(48):
            have = [s for s in range(NSEED)
                    if (i, +1, d, s) in ok and (i, -1, d, s) in ok]
            if not have:
                # no usable leg for this gain at this step: contribute a zero
                # column rather than silently shrinking the parameter space
                Jc.append(np.zeros_like(base_c))
                Jr.append(np.zeros(T * base_j.shape[1]))
                norms.append(0.0)
                dropped += 1
                continue
            cp = np.mean([ok[(i, +1, d, s)][0] for s in have], axis=0)
            cm = np.mean([ok[(i, -1, d, s)][0] for s in have], axis=0)
            Jc.append((cp - cm) / (2 * d))
            jp = np.mean([ok[(i, +1, d, s)][1][:T] for s in have], axis=0)
            jm = np.mean([ok[(i, -1, d, s)][1][:T] for s in have], axis=0)
            Jr.append(((jp - jm) / (2 * d)).flatten())
            # the proportionality test: whitened one-sided response against the step
            norms.append(float(np.linalg.norm((jp - base_j) / sj) / np.sqrt(jp.size)))
        Jc, Jr = np.array(Jc).T, np.array(Jr).T
        Hc = (Jc * wc[:, None]).T @ (Jc * wc[:, None])
        w = np.tile(sj, T)[:, None]
        Hr = (Jr / w).T @ (Jr / w)
        c90, c99, cpr = eff_dim(Hc)
        r90, r99, rpr = eff_dim(Hr)
        mn = float(np.median(norms))
        rows.append({"delta": d, "n_gains_with_no_usable_leg": dropped,
                     "median_response_norm": mn,
                     "response_over_delta": mn / d,
                     "coarse": {"eff_dim_90": c90, "eff_dim_99": c99, "participation_ratio": cpr},
                     "rich": {"eff_dim_90": r90, "eff_dim_99": r99, "participation_ratio": rpr}})
        print(f"  delta {d:.2f}: dropped {dropped}/48  |dObs| {mn:.4f}  ratio {mn/d:8.4f}  "
              f"coarse {c90}/{c99}  rich {r90}/{r99}", flush=True)

    ratios = [r["response_over_delta"] for r in rows]
    valid = [r for r in rows if r["delta"] >= 0.25]
    out = {
        "experiment": "AB4_flygym_step_size",
        "question": "does the FlyGym effective dimension survive at a step where the "
                    "response is proportional, as S1.5 requires?",
        "n_params": 48, "n_seeds": NSEED, "rollout_seconds": RUN_S,
        "frames_kept": int(T), "n_rich_observables": int(T * 42),
        "deltas": DELTAS, "by_delta": rows,
        "n_legs_diverged": len(diverged),
        "n_legs_total": len(res),
        "ratio_spread_all_steps": max(ratios) / min(ratios),
        "ratio_spread_valid_region": (max(r["response_over_delta"] for r in valid) /
                                      min(r["response_over_delta"] for r in valid)),
        "published_at_delta_0.05": {"coarse_eff_dim_90_99": [1, 1],
                                    "rich_eff_dim_90_99": [16, 38]},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("saved ->", OUT, flush=True)
    print(f"ratio spread over all steps {out['ratio_spread_all_steps']:.2f}, "
          f"over the valid region {out['ratio_spread_valid_region']:.2f} "
          f"(1 = perfectly proportional)")


if __name__ == "__main__":
    main()
