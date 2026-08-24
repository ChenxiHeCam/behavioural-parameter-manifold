"""AB2: does the pyloric effective dimension depend on how many valid points
were probed, or on how many observables the circuit is read through?

The deposited figure, 2.13 of 31, is the mean over K=8 rhythm-gated points. Two
things could make that number an artefact rather than a measurement:

  (i)  K=8 is a small sample of a population whose per-point values span
       1.10-3.87, so the mean carries an unreported sampling error;
  (ii) 15 summary statistics cannot resolve more than 15 of 31 directions, the
       same rank cap that inflated the FlyGym reading.

So every point is probed twice, through the 15 summary statistics and through
subsampled voltage traces (thousands of observables, in excess of the 31
parameters), and the full distribution is reported against K rather than a
single mean. A K-scan that flattens is evidence the number is a property of the
population; one that drifts is evidence it is a property of the sample.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# the pyloric simulator is a Cython extension and is not fork-safe
CTX = mp.get_context("spawn")

PARAMS = "/root/autodl-tmp/cell/e1_valid_params.npy"
OUT = "/root/autodl-tmp/cell/AB2_stg_scan.json"
DELTA = 0.05
SUB = 400              # voltage subsampling stride -> ~1100 obs per cell
K_MAX = int(os.environ.get("KMAX", "48"))
NW = int(os.environ.get("NW", "30"))
SEED = 0


def _cols():
    from pyloric import create_prior
    return create_prior().sample((1,)).columns


def _sim(vals):
    """One simulation. Returns (15 summary statistics, subsampled voltage)."""
    import numpy as np
    from pyloric import simulate, summary_stats
    import pandas as pd
    row = pd.Series(np.asarray(vals, dtype=float), index=_COLS)
    out = simulate(row)
    v = np.asarray(out["voltage"], dtype=np.float32)
    ss = np.asarray(summary_stats(out), dtype=float).ravel()
    return ss, v[:, ::SUB]


_COLS = None


def _init():
    global _COLS
    _COLS = _cols()


def leg(a):
    """One finite-difference leg: parameter idx scaled by (1 +- delta)."""
    k, idx, sign, th = a
    th = np.array(th, dtype=float)
    if idx >= 0:
        th[idx] *= (1.0 + sign * DELTA)
    try:
        ss, v = _sim(th)
        if not np.all(np.isfinite(ss)):
            return (k, idx, sign), None
        return (k, idx, sign), (ss.astype(np.float32), v)
    except Exception:
        return (k, idx, sign), None


def gauss_newton(cols, base):
    """Whitened Gauss-Newton curvature from finite-difference columns."""
    J = np.array(cols).T
    sd = np.abs(base)
    sd = np.where(sd > 1e-9, sd, np.nanmedian(sd[sd > 1e-9]) if np.any(sd > 1e-9) else 1.0)
    Jw = J / sd[:, None]
    H = Jw.T @ Jw
    return H if np.all(np.isfinite(H)) else None


def eff(H):
    ev = np.sort(np.maximum(np.linalg.eigvalsh(H), 0))[::-1]
    if ev.sum() <= 0:
        return None
    cs = np.cumsum(ev) / ev.sum()
    pos = ev[ev > ev.max() * 1e-14]
    return {"eff_dim_90": int(np.searchsorted(cs, 0.90) + 1),
            "eff_dim_99": int(np.searchsorted(cs, 0.99) + 1),
            "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "spectrum_decades": float(np.log10(pos.max() / pos.min())) if len(pos) > 1 else 0.0}


def main():
    t0 = time.time()
    P = np.load(PARAMS)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(len(P), size=min(K_MAX, len(P)), replace=False)
    pts = P[sel]
    print(f"{len(pts)} valid points x (1 + 2*31) legs = {len(pts)*63} simulations", flush=True)

    jobs = []
    for k in range(len(pts)):
        jobs.append((k, -1, 0, pts[k]))
        for i in range(31):
            for s in (+1, -1):
                jobs.append((k, i, s, pts[k]))

    res, done = {}, 0
    with ProcessPoolExecutor(max_workers=NW, mp_context=CTX, initializer=_init) as ex:
        for key, val in ex.map(leg, jobs, chunksize=4):
            res[key] = val
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    per_point = []
    for k in range(len(pts)):
        b = res.get((k, -1, 0))
        if b is None:
            per_point.append({"k": k, "status": "baseline_failed"})
            continue
        b_ss, b_v = b
        rec = {"k": k, "status": "ok"}
        for tag, pick, nobs in (("coarse", 0, len(b_ss)), ("rich", 1, b_v.size)):
            cols, ok = [], True
            for i in range(31):
                p, m = res.get((k, i, +1)), res.get((k, i, -1))
                if p is None or m is None:
                    ok = False
                    break
                a, c = p[pick], m[pick]
                a, c = np.asarray(a, float).ravel(), np.asarray(c, float).ravel()
                n = min(a.size, c.size)
                cols.append((a[:n] - c[:n]) / (2 * DELTA))
            if not ok:
                rec[tag] = {"status": "leg_failed"}
                continue
            n = min(c.size for c in cols)
            cols = [c[:n] for c in cols]
            base = (b_ss if pick == 0 else np.asarray(b_v, float).ravel())[:n]
            H = gauss_newton(cols, base)
            e = eff(H) if H is not None else None
            rec[tag] = ({"status": "ok", "n_obs": int(n), **e} if e else {"status": "degenerate"})
        per_point.append(rec)

    def scan(tag):
        """Running mean of eff_dim_90 against K: does it settle?"""
        vals = [r[tag]["eff_dim_90"] for r in per_point
                if r.get("status") == "ok" and r.get(tag, {}).get("status") == "ok"]
        out = []
        for K in (4, 8, 12, 16, 24, 32, 48):
            if K > len(vals):
                break
            v = np.array(vals[:K], float)
            out.append({"K": K, "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if K > 1 else 0.0,
                        "sem": float(v.std(ddof=1) / np.sqrt(K)) if K > 1 else 0.0})
        return {"per_point_values": vals, "running": out,
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else None,
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}

    result = {
        "experiment": "AB2_stg_scaling_scan",
        "question": "is the pyloric effective dimension a property of the population or of K=8?",
        "n_points_requested": int(K_MAX), "delta": DELTA, "voltage_stride": SUB,
        "per_point": per_point,
        "coarse_15_summary_stats": scan("coarse"),
        "rich_voltage_observables": scan("rich"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(result, open(OUT, "w"), indent=1)
    print("saved ->", OUT, f"({result['elapsed_sec']}s)", flush=True)
    for t in ("coarse_15_summary_stats", "rich_voltage_observables"):
        s = result[t]
        print(f"{t}: n={s['n']} mean={s['mean']} sd={s['sd']}")
        for r in s["running"]:
            print(f"   K={r['K']:3d}  {r['mean']:.2f} +- {r['sem']:.2f}")


if __name__ == "__main__":
    main()
