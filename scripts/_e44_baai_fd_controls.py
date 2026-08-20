"""E-44 to E-46: the controls the BAAIWorm measurement needs to be defensible.

E-44 noise null. The 3076-weight Jacobian gave an effective dimension of 8 (90%).
Some of that could be finite-difference noise. NEURON is deterministic, so the null
is built by pairing repeat simulations that differ only in an irrelevant setting
(simulation duration truncated to a common window), producing columns of the same
magnitude as the real ones but carrying no parameter information.

E-45 named mechanisms with the rich observable. The deposited per-channel result
uses six behavioural scalars, so its rank cannot exceed six; the biological reading
-- wiring stiff, fast calcium sloppy -- rests on that. Here the same named
mechanisms are re-measured against the full motor-neuron trajectory.

E-46 step-size scan, so the working point is shown to be in the linear regime.
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
TSTOP = 1500
DELTA = 0.5
N_SAMPLE = 400
SCAN = [0.05, 0.1, 0.25, 0.5, 1.0]


def call(spec):
    try:
        p = subprocess.run([PY, WORKER, json.dumps(spec)],
                           capture_output=True, text=True, timeout=900)
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                return np.array(json.loads(ln[7:])["motor"], dtype=np.float32)
    except Exception:
        pass
    return None


def job(a):
    return a[:-1], call(a[-1])


def eff(H):
    ev = np.sort(np.linalg.eigvalsh(H))[::-1]
    ev = np.maximum(ev, 0)
    if ev.sum() <= 0:
        return 0, 0, 0.0
    cs = np.cumsum(ev) / ev.sum()
    return (int(np.searchsorted(cs, 0.90) + 1),
            int(np.searchsorted(cs, 0.99) + 1),
            float(ev.sum() ** 2 / (ev ** 2).sum()))


def hessian(store, key_fn, ids, base):
    sd = base.std(1).clip(1e-3)[:, None]
    cols = []
    for i in ids:
        p, q = store.get(key_fn(i, +1)), store.get(key_fn(i, -1))
        if p is None or q is None:
            return None, None
        t = min(base.shape[1], p.shape[1], q.shape[1])
        cols.append((((p[:, :t] - q[:, :t]) / (2 * DELTA)) / sd).ravel())
    n = min(len(c) for c in cols)
    J = np.array([c[:n] for c in cols], dtype=np.float64).T
    return J.T @ J, np.linalg.norm(J, axis=0)


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    ids = sorted(rng.choice(3076, N_SAMPLE, replace=False).tolist())
    out = {}

    # ---------------- E-46 step-size scan -----------------------------------
    scan_specs = [("base", 0, 0, {"conn": -1, "tstop": TSTOP})]
    for d in SCAN:
        for i in ids[:20]:
            for sg in (+1, -1):
                scan_specs.append((f"d{d}", i, sg,
                                   {"conn": i, "sign": sg, "delta": d,
                                    "tstop": TSTOP}))
    print(f"E-46: {len(scan_specs)} rollouts for the step scan", flush=True)
    st = {}
    with ProcessPoolExecutor(max_workers=40, mp_context=CTX) as pool:
        for k, v in pool.map(job, scan_specs, chunksize=1):
            st[k] = v
    base = st[("base", 0, 0)]
    sd = base.std(1).clip(1e-3)[:, None]
    scan_rows = []
    for d in SCAN:
        vals = []
        for i in ids[:20]:
            p, q = st.get((f"d{d}", i, +1)), st.get((f"d{d}", i, -1))
            if p is None or q is None:
                continue
            t = min(base.shape[1], p.shape[1], q.shape[1])
            vals.append(float(np.sqrt(((((p[:, :t] - q[:, :t]) / sd)) ** 2).mean())))
        m = float(np.median(vals)) if vals else float("nan")
        scan_rows.append({"delta": d, "median_rel_rms": m,
                          "rel_rms_over_delta": m / d})
        print(f"  delta {d:<6} rel_rms {m:.4e}  ratio {m/d:.4e}", flush=True)
    out["E46_step_scan"] = scan_rows

    # ---------------- E-44 noise null ---------------------------------------
    # repeat the baseline under settings that should not matter, and use the
    # residual spread as the magnitude of a parameter-free Jacobian
    rep_specs = [(f"rep{r}", -1, 0, {"conn": -1, "tstop": TSTOP + r})
                 for r in range(24)]
    print(f"E-44: {len(rep_specs)} repeat baselines", flush=True)
    rp = {}
    with ProcessPoolExecutor(max_workers=24, mp_context=CTX) as pool:
        for k, v in pool.map(job, rep_specs, chunksize=1):
            rp[k] = v
    reps = [v for v in rp.values() if v is not None]
    if len(reps) >= 4:
        T = min(v.shape[1] for v in reps)
        A = np.array([v[:, :T] for v in reps], dtype=float)
        rr = np.random.default_rng(1)
        cols = []
        for _ in range(N_SAMPLE):
            a = rr.integers(0, len(reps))
            b = rr.integers(0, len(reps))
            cols.append((((A[a] - A[b]) / (2 * DELTA)) / sd[:, :1]).ravel())
        n = min(len(c) for c in cols)
        Jn = np.array([c[:n] for c in cols]).T
        e90, e99, pr = eff(Jn.T @ Jn)
        out["E44_noise_null"] = {"n_repeats": len(reps), "eff_dim_90": e90,
                                 "eff_dim_99": e99, "participation_ratio": pr}
        print(f"  noise null: eff-dim {e90}/{e99} of {N_SAMPLE}  PR {pr:.2f}",
              flush=True)
    else:
        out["E44_noise_null"] = {"note": "insufficient repeats"}

    json.dump(out, open("/root/autodl-tmp/E44_baai_controls.json", "w"), indent=2)
    print("=== E-44/E-46 DONE %.0fs ===" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
