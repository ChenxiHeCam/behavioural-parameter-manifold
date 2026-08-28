"""E-3 stage 2, run standalone from the gated points.

Stage 1 completed the sweep and saved 151 valid points. Stage 2 kept losing its
results because a whole pool block would die and take its store with it, so here
each point is handled independently, its Hessian assembled as soon as its own
62 simulations return, and a failed point costs only that point.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/root/autodl-tmp")
from _e3_pyloric_gate import _sim, eff_dim, DELTA

CTX = mp.get_context("spawn")
K_MAX = 60


def one_leg(a):
    idx, sign, th = a
    th = np.array(th, dtype=float)
    if idx >= 0:
        th[idx] *= (1.0 + sign * DELTA)
    try:
        ss, v = _sim(th)
        return idx, sign, ss, v.astype(np.float32)
    except Exception:
        return idx, sign, None, None


def hessians_for_point(th, pool):
    legs = [(-1, 0, th)] + [(i, s, th) for i in range(31) for s in (+1, -1)]
    res = {}
    for idx, sign, ss, v in pool.map(one_leg, legs, chunksize=1):
        res[(idx, sign)] = (ss, v)
    b = res.get((-1, 0))
    if b is None or b[1] is None:
        return None
    bss, bv = b
    T = bv.shape[1]
    Jc, Jr, used = [], [], []
    for i in range(31):
        p, m = res.get((i, +1)), res.get((i, -1))
        pv = p[1] if p is not None else None
        mv = m[1] if m is not None else None
        # a perturbed conductance can make the integrator diverge; fall back to a
        # one-sided difference against the base rather than discarding the point
        if pv is not None and mv is not None:
            t = min(T, pv.shape[1], mv.shape[1])
            dv = (pv[:, :t] - mv[:, :t]) / (2 * DELTA)
            ds = ((p[0] - m[0]) / (2 * DELTA)
                  if p[0] is not None and m[0] is not None else None)
        elif pv is not None:
            t = min(T, pv.shape[1])
            dv = (pv[:, :t] - bv[:, :t]) / DELTA
            ds = (p[0] - bss) / DELTA if p[0] is not None else None
        elif mv is not None:
            t = min(T, mv.shape[1])
            dv = (bv[:, :t] - mv[:, :t]) / DELTA
            ds = (bss - m[0]) / DELTA if m[0] is not None else None
        else:
            continue
        used.append(i)
        Jc.append(ds if ds is not None else np.full(bss.shape, np.nan))
        Jr.append(dv.astype(float).ravel())
    if len(used) < 20:
        return None
    t3 = min(len(x) for x in Jr)
    Tm = t3 // 3
    Jc = np.array(Jc).T
    Jr = np.array([x[:3 * Tm] for x in Jr]).T
    out = {}
    if np.all(np.isfinite(Jc)) and np.all(np.isfinite(bss)):
        w = 1.0 / np.maximum(np.abs(bss), 1e-8)
        out["coarse"] = eff_dim(np.linalg.eigvalsh((Jc * w[:, None]).T @ (Jc * w[:, None])))
    sv = bv[:, :Tm].std(1).clip(1e-3)
    wr = np.repeat(1.0 / sv, Tm)
    Hr = (Jr * wr[:, None]).T @ (Jr * wr[:, None])
    if np.all(np.isfinite(Hr)):
        out["rich"] = eff_dim(np.linalg.eigvalsh(Hr))
    out["n_obs_rich"] = int(3 * Tm)
    out["n_params_used"] = len(used)
    return out


def main():
    t0 = time.time()
    pts = json.load(open("/root/autodl-tmp/E3_valid_points.json"))
    K = min(K_MAX, len(pts))
    sel = [pts[i] for i in np.linspace(0, len(pts) - 1, K).astype(int)]
    print(f"probing {K} of {len(pts)} valid points", flush=True)

    coarse, rich, nobs, used_counts = [], [], 0, []
    for k, th in enumerate(sel):
        try:
            with ProcessPoolExecutor(max_workers=32, mp_context=CTX) as pool:
                r = hessians_for_point(th, pool)
        except Exception as e:
            print(f"  point {k} failed ({type(e).__name__})", flush=True)
            continue
        if r is None:
            print(f"  point {k} incomplete", flush=True)
            continue
        used_counts.append(r.get("n_params_used", 31))
        if "coarse" in r:
            coarse.append(r["coarse"])
        if "rich" in r:
            rich.append(r["rich"])
        nobs = r["n_obs_rich"]
        if (k + 1) % 5 == 0:
            print(f"  {k+1}/{K} points, coarse {len(coarse)} rich {len(rich)}, "
                  f"{time.time()-t0:.0f}s", flush=True)

    def summarise(rows, n):
        if not rows:
            return {"n_points": 0, "n_observables": n}
        a = np.array(rows, dtype=float)
        return {"n_points": len(rows), "n_observables": n,
                "eff_dim_90_mean": float(a[:, 0].mean()),
                "eff_dim_90_sd": float(a[:, 0].std()),
                "eff_dim_90_min": int(a[:, 0].min()),
                "eff_dim_90_max": int(a[:, 0].max()),
                "eff_dim_99_mean": float(a[:, 1].mean()),
                "eff_dim_99_sd": float(a[:, 1].std()),
                "participation_ratio_mean": float(a[:, 2].mean()),
                "participation_ratio_sd": float(a[:, 2].std()),
                "per_point_eff_dim_90": [int(x) for x in a[:, 0]]}

    out = {"experiment": "E3_pyloric_stage2",
           "n_valid_points_available": len(pts), "n_probed": K,
           "n_params": 31, "delta": DELTA,
           "params_usable_per_point_mean": float(np.mean(used_counts)) if used_counts else 0,
           "coarse": summarise(coarse, 15),
           "rich": summarise(rich, nobs),
           "published": {"eff_dim_mean": 2.13, "K": 8,
                         "note": "mean of eight points, coarse observables"},
           "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(out, open("/root/autodl-tmp/E3_pyloric_stage2.json", "w"), indent=2)
    print("\n=== E-3 RESULT ===", flush=True)
    for tag in ("coarse", "rich"):
        s = out[tag]
        if s["n_points"]:
            print(f"{tag:7s} {s['n_observables']:5d} obs, {s['n_points']:3d} points: "
                  f"eff-dim90 {s['eff_dim_90_mean']:.2f} +-{s['eff_dim_90_sd']:.2f} "
                  f"(range {s['eff_dim_90_min']}-{s['eff_dim_90_max']}) of 31",
                  flush=True)
    print("E3S2_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
