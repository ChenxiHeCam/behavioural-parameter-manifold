"""E-7 and E-11 on the real modWorm model, rebuilt from the public repository.

E-7  The union of behaviour-specific identifiable subspaces is reported as growing
     with the repertoire, but the deposited numbers are non-monotone: four
     behaviours give a union of 5 and twelve give 4, because the two figures come
     from different behaviour sets. Here one nested sequence is used, adding one
     behaviour at a time to the same set, so the curve is a curve of one quantity.
     Both the 90% and 99% thresholds are reported, since the published growth
     exists only at 99%.

E-11 The identity of the stiff mechanisms is reported at a single point. The
     deposited five-point control shows the ordering reversing between points
     (the sigmoid gain, called sloppy throughout, ranks stiffest at one of them).
     Here the Hessian is computed at many points on the manifold and the rank
     correlation of the mechanism ordering between points is reported, which is
     what the biophysical conclusion actually rests on.

Both use the rich posture observable (tangent-angle trajectory), so neither is
capped by the six coarse scalars.
"""
import os
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'   # 172 processes each opening a BLAS pool oversubscribes the box
import json, os, subprocess, sys, time
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

CTX = mp.get_context("spawn")
PY = "/root/miniconda3/bin/python"
WORKER = "/root/autodl-tmp/mw_rollout.py"
MECH = ["gap", "syn", "leak", "Cm", "rise", "fall", "B"]
BEHAV = ["input_mat_gentle_ant_touch", "input_mat_gentle_post_touch",
         "input_mat_harsh_ant_touch", "input_mat_harsh_post_touch"]
DELTA = 0.25
NSTEP = 150
N_POINTS = 24          # points sampled on the manifold for E-11
SIGMA = 0.15           # spread of those points around the reference parameters


def call(spec):
    try:
        p = subprocess.run([PY, WORKER, json.dumps(spec)],
                           capture_output=True, text=True, timeout=900)
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                d = json.loads(ln[7:])
                return np.array(d["phi"], dtype=float)
    except Exception:
        pass
    return None


def job(a):
    return a, call(a[-1])


def hessian(base_scale, behaviour):
    """Per-mechanism Gauss-Newton curvature at one point, one behaviour."""
    specs = {}
    for m in MECH:
        for sg in (+1, -1):
            s = dict(base_scale)
            s[m] = s.get(m, 1.0) * (1 + sg * DELTA)
            s["stim"] = behaviour
            s["n_steps"] = NSTEP
            specs[(m, sg)] = s
    b = dict(base_scale)
    b["stim"] = behaviour
    b["n_steps"] = NSTEP
    specs[("base", 0)] = b
    return specs


def eff_dim(H):
    ev = np.sort(np.linalg.eigvalsh(H))[::-1]
    ev = np.maximum(ev, 0)
    if ev.sum() <= 0:
        return 0, 0, 0.0
    cs = np.cumsum(ev) / ev.sum()
    return (int(np.searchsorted(cs, 0.90) + 1), int(np.searchsorted(cs, 0.99) + 1),
            float(ev.sum() ** 2 / (ev ** 2).sum()))


def build(store, base_key, behaviour, point):
    b = store.get((point, behaviour, "base", 0))
    if b is None:
        return None, None
    T = b.shape[0]
    cols = []
    for m in MECH:
        p = store.get((point, behaviour, m, +1))
        q = store.get((point, behaviour, m, -1))
        if p is None or q is None:
            return None, None
        t = min(T, p.shape[0], q.shape[0])
        cols.append(((p[:t] - q[:t]) / (2 * DELTA)).ravel())
    t = min(len(c) for c in cols)
    J = np.array([c[:t] for c in cols]).T
    sc = b[:T].std(0).clip(1e-6)
    w = np.tile(sc, t // len(sc) + 1)[:t]
    Jw = J / w[:, None]
    return Jw.T @ Jw, np.linalg.norm(Jw, axis=0)


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    points = [{}]                                   # reference point first
    for _ in range(N_POINTS - 1):
        points.append({m: float(np.exp(rng.normal(0, SIGMA))) for m in MECH})

    specs = []
    for pi, pt in enumerate(points):
        for bh in BEHAV:
            for k, s in hessian(pt, bh).items():
                specs.append((pi, bh, k[0], k[1], s))
    print(f"{len(specs)} rollouts", flush=True)

    store = {}
    nw = 40
    with ProcessPoolExecutor(max_workers=nw, mp_context=CTX) as ex:
        for n, (a, phi) in enumerate(ex.map(job, specs, chunksize=1), 1):
            store[(a[0], a[1], a[2], a[3])] = phi
            if n % 100 == 0:
                print(f"  {n}/{len(specs)}  {time.time()-t0:.0f}s", flush=True)

    # ---------------- E-11: stability of the stiff ordering across points -----
    orders, dims = [], []
    for pi in range(len(points)):
        H, norms = build(store, None, BEHAV[1], pi)
        if H is None:
            continue
        e90, e99, pr = eff_dim(H)
        dims.append((e90, e99, pr))
        orders.append(list(np.argsort(norms)[::-1]))
    from scipy.stats import spearmanr
    rhos = []
    for a, b in itertools.combinations(range(len(orders)), 2):
        ra = np.empty(7); ra[orders[a]] = np.arange(7)
        rb = np.empty(7); rb[orders[b]] = np.arange(7)
        rhos.append(float(spearmanr(ra, rb).statistic))
    top1 = [MECH[o[0]] for o in orders]

    # ---------------- E-7: nested union over one behaviour set ----------------
    curve = []
    for k in range(1, len(BEHAV) + 1):
        Hs = []
        for bh in BEHAV[:k]:
            H, _ = build(store, None, bh, 0)
            if H is not None:
                Hs.append(H / max(np.trace(H), 1e-30))
        if not Hs:
            continue
        U = np.sum(Hs, axis=0)
        e90, e99, pr = eff_dim(U)
        curve.append({"n_behaviours": k, "eff_dim_90": e90, "eff_dim_99": e99,
                      "participation_ratio": pr})
        print(f"  nested union {k} behaviours -> {e90}/{e99} of 7  PR {pr:.2f}",
              flush=True)

    out = {
        "experiment": "E7_E11_modworm",
        "build": "modWorm public repository, Cook connectome, 279 neurons",
        "observable": "body tangent-angle trajectory (rich)",
        "delta": DELTA, "n_steps": NSTEP,
        "E7_nested_union": {"behaviour_sequence": BEHAV, "curve": curve,
                            "note": "one nested set, so the union cannot decrease"},
        "E11_point_dependence": {
            "n_points": len(orders), "sigma": SIGMA,
            "eff_dim_per_point_90": [d[0] for d in dims],
            "eff_dim_per_point_99": [d[1] for d in dims],
            "stiffest_mechanism_per_point": top1,
            "pairwise_order_spearman_mean": float(np.mean(rhos)) if rhos else None,
            "pairwise_order_spearman_sd": float(np.std(rhos)) if rhos else None,
            "fraction_of_points_with_same_stiffest": float(
                max(top1.count(m) for m in set(top1)) / len(top1)) if top1 else None,
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open("/root/autodl-tmp/E7_E11_modworm.json", "w"), indent=2)
    print("\n=== E-7 / E-11 RESULT ===", flush=True)
    for c in curve:
        print(f"  union over {c['n_behaviours']} behaviours: "
              f"{c['eff_dim_90']}/{c['eff_dim_99']} of 7", flush=True)
    d = out["E11_point_dependence"]
    print(f"  stiff ordering across {d['n_points']} points: "
          f"mean Spearman {d['pairwise_order_spearman_mean']}", flush=True)
    print(f"  stiffest mechanism agrees at "
          f"{100*d['fraction_of_points_with_same_stiffest']:.0f}% of points", flush=True)
    print("E7_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
