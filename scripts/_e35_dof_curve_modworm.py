"""E-35: how many degrees of freedom does behaviour constrain, as behaviours are added?

This is the paper's central measurement stated as one quantity. For a repertoire of
n behaviours, and for every subset size k = 1..n, the union Gauss-Newton curvature
over that subset is formed and its effective dimension recorded. Averaging over all
subsets of each size removes the arbitrariness of a single nested ordering, which
the published curve did not do.

Two things are done differently from the deposited version. Observables are the
full posture trajectory rather than six summary scalars, so the count is not capped
by the observable rank. And the curve is reported at 90 and 99 per cent of spectral
mass and by the threshold-free participation ratio, because the published growth
appeared only at 99 per cent.

A shuffled control accompanies it: the same curve built from Jacobians whose
behaviour labels have been permuted, which is what union growth looks like when the
behaviours carry no distinct information.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import itertools, json, subprocess, sys, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

CTX = mp.get_context("spawn")
PY = "/root/miniconda3/bin/python"
WORKER = "/root/autodl-tmp/mw_rollout.py"
MECH = ["gap", "syn", "leak", "Cm", "rise", "fall", "B"]
BEHAV = ["input_mat_gentle_ant_touch", "input_mat_gentle_post_touch",
         "input_mat_harsh_ant_touch", "input_mat_harsh_post_touch"]
NSTEP = 150
DELTA = 0.25
BASE = {m: 1.0 for m in MECH}


def rollout(a):
    scale, behaviour = a
    spec = dict(scale)
    spec["stim"] = behaviour
    spec["n_steps"] = NSTEP
    try:
        p = subprocess.run([PY, WORKER, json.dumps(spec)],
                           capture_output=True, text=True, timeout=600)
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                return np.array(json.loads(ln[7:])["phi"], dtype=float)
    except Exception:
        pass
    return None


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
    jobs = []
    for b in BEHAV:
        jobs.append((dict(BASE), b))
        for m in MECH:
            for sg in (+1, -1):
                s = dict(BASE)
                s[m] = 1.0 + sg * DELTA
                jobs.append((s, b))
    print(f"{len(jobs)} rollouts", flush=True)

    with ProcessPoolExecutor(max_workers=40, mp_context=CTX) as pool:
        outs = list(pool.map(rollout, jobs, chunksize=1))
    print(f"rollouts done {time.time()-t0:.0f}s", flush=True)

    # one normalised curvature matrix per behaviour
    H = {}
    Jstore = {}
    idx = 0
    for b in BEHAV:
        base = outs[idx]; idx += 1
        cols = []
        for _m in MECH:
            p = outs[idx]; idx += 1
            q = outs[idx]; idx += 1
            if base is None or p is None or q is None:
                cols.append(None); continue
            t = min(base.shape[0], p.shape[0], q.shape[0])
            sd = base[:t].std(0).clip(1e-6)
            cols.append((((p[:t] - q[:t]) / (2 * DELTA)) / sd).ravel())
        if any(c is None for c in cols):
            print(f"  behaviour {b} incomplete", flush=True)
            continue
        n = min(len(c) for c in cols)
        J = np.array([c[:n] for c in cols]).T
        Jstore[b] = J
        M = J.T @ J
        H[b] = M / max(np.trace(M), 1e-30)

    avail = list(H)
    print(f"{len(avail)} behaviours usable", flush=True)
    for b in avail:
        e90, e99, pr = eff(H[b])
        print(f"  {b}: alone {e90}/{e99} of 7  PR {pr:.2f}", flush=True)

    curve = []
    for k in range(1, len(avail) + 1):
        rec = []
        for combo in itertools.combinations(avail, k):
            U = sum(H[b] for b in combo)
            rec.append(eff(U))
        a = np.array(rec, dtype=float)
        curve.append({"n_behaviours": k, "n_subsets": len(rec),
                      "eff_dim_90_mean": float(a[:, 0].mean()),
                      "eff_dim_90_sd": float(a[:, 0].std()),
                      "eff_dim_99_mean": float(a[:, 1].mean()),
                      "eff_dim_99_sd": float(a[:, 1].std()),
                      "participation_ratio_mean": float(a[:, 2].mean()),
                      "participation_ratio_sd": float(a[:, 2].std())})
        c = curve[-1]
        print(f"  k={k} ({len(rec)} subsets): 90% {c['eff_dim_90_mean']:.2f}"
              f"+-{c['eff_dim_90_sd']:.2f}  99% {c['eff_dim_99_mean']:.2f}"
              f"+-{c['eff_dim_99_sd']:.2f}  PR {c['participation_ratio_mean']:.2f}",
              flush=True)

    # shuffled control: rows of each behaviour's Jacobian permuted independently,
    # which destroys the behaviour-specific structure but keeps the magnitudes
    rng = np.random.default_rng(0)
    Hs = {}
    for b in avail:
        J = Jstore[b].copy()
        for j in range(J.shape[1]):
            rng.shuffle(J[:, j])
        M = J.T @ J
        Hs[b] = M / max(np.trace(M), 1e-30)
    shuf = []
    for k in range(1, len(avail) + 1):
        rec = [eff(sum(Hs[b] for b in c))
               for c in itertools.combinations(avail, k)]
        a = np.array(rec, dtype=float)
        shuf.append({"n_behaviours": k,
                     "eff_dim_90_mean": float(a[:, 0].mean()),
                     "eff_dim_99_mean": float(a[:, 1].mean()),
                     "participation_ratio_mean": float(a[:, 2].mean())})

    out = {"experiment": "E35_dof_vs_behaviours",
           "model": "real modWorm, public build, 7 mechanisms",
           "observable": "body tangent-angle trajectory (rich, uncapped)",
           "delta": DELTA, "n_steps": NSTEP,
           "curve": curve, "shuffled_control": shuf,
           "note": "each point averages over all subsets of that size, so the curve "
                   "does not depend on one arbitrary ordering of behaviours",
           "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(out, open("/root/autodl-tmp/E35_dof_curve.json", "w"), indent=2)
    print("\n=== E-35 RESULT: constrained degrees of freedom ===", flush=True)
    for c, s in zip(curve, shuf):
        print(f"  {c['n_behaviours']} behaviour(s): "
              f"90% {c['eff_dim_90_mean']:.2f}+-{c['eff_dim_90_sd']:.2f}   "
              f"99% {c['eff_dim_99_mean']:.2f}+-{c['eff_dim_99_sd']:.2f}   "
              f"PR {c['participation_ratio_mean']:.2f}   "
              f"[shuffled 90% {s['eff_dim_90_mean']:.2f}]", flush=True)
    print("E35_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
