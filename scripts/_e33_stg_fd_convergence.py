"""E-33: is the pyloric rich-observable dimension of 25 of 31 signal or noise?

Measuring the circuit through 6600 voltage observables instead of 15 summary
statistics raises the effective dimension from 2.4 to 25.4. With that many
observables and a finite-difference Jacobian, some of the rise could be numerical
noise, exactly as the FlyGym control showed (16/38 against a noise floor of 7/22).

The pyloric simulator is deterministic, so a repeat-simulation null is not
available. Instead the standard convergence test is used: the Hessian is computed
at three finite-difference step sizes. A direction carrying real curvature is
reproduced as the step shrinks; a direction that is finite-difference noise scales
as 1/delta^2 and its eigenvector is unstable. Two quantities are reported per
point: the effective dimension at each step, and the principal-angle agreement
between the leading subspaces obtained at successive steps.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json, sys, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/root/autodl-tmp")
from e3_pyloric import _sim

CTX = mp.get_context("spawn")
DELTAS = [0.10, 0.05, 0.025]
K_POINTS = 12


def leg(a):
    idx, sign, delta, th = a
    th = np.array(th, dtype=float)
    if idx >= 0:
        th[idx] *= (1.0 + sign * delta)
    try:
        _, v = _sim(th)
        return idx, sign, delta, v.astype(np.float32)
    except Exception:
        return idx, sign, delta, None


def spectrum(res, delta, base):
    T = base.shape[1]
    cols = []
    for i in range(31):
        p, m = res.get((i, +1, delta)), res.get((i, -1, delta))
        if p is None or m is None:
            return None, None
        t = min(T, p.shape[1], m.shape[1])
        cols.append(((p[:, :t] - m[:, :t]) / (2 * delta)).astype(float).ravel())
    n = min(len(c) for c in cols)
    Tm = n // 3
    J = np.array([c[:3 * Tm] for c in cols]).T
    sd = base[:, :Tm].std(1).clip(1e-3)
    w = np.repeat(1.0 / sd, Tm)
    H = (J * w[:, None]).T @ (J * w[:, None])
    if not np.all(np.isfinite(H)):
        return None, None
    ev, V = np.linalg.eigh(H)
    o = np.argsort(ev)[::-1]
    return ev[o], V[:, o]


def eff(ev):
    ev = np.maximum(ev, 0)
    if ev.sum() <= 0:
        return 0, 0.0
    cs = np.cumsum(ev) / ev.sum()
    return int(np.searchsorted(cs, 0.90) + 1), float(ev.sum() ** 2 / (ev ** 2).sum())


def subspace_cos(A, B, k):
    """Mean cosine of principal angles between two leading k-subspaces."""
    s = np.linalg.svd(A[:, :k].T @ B[:, :k], compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1)))


def main():
    t0 = time.time()
    pts = json.load(open("/root/autodl-tmp/E3_valid_points.json"))
    sel = [pts[i] for i in np.linspace(0, len(pts) - 1, K_POINTS).astype(int)]
    rows = []
    with ProcessPoolExecutor(max_workers=36, mp_context=CTX) as pool:
        for k, th in enumerate(sel):
            jobs = [(-1, 0, DELTAS[0], th)]
            for d in DELTAS:
                for i in range(31):
                    jobs += [(i, +1, d, th), (i, -1, d, th)]
            try:
                out = list(pool.map(leg, jobs, chunksize=1))
            except Exception as e:
                print(f"  point {k} pool failure ({type(e).__name__})", flush=True)
                continue
            res, base = {}, None
            for idx, sign, d, v in out:
                if idx < 0:
                    base = v
                else:
                    res[(idx, sign, d)] = v
            if base is None:
                print(f"  point {k} no baseline", flush=True)
                continue
            row = {"point": k, "per_delta": {}}
            spec, vecs = {}, {}
            ok = True
            for d in DELTAS:
                ev, V = spectrum(res, d, base)
                if ev is None:
                    ok = False
                    break
                e90, pr = eff(ev)
                spec[d], vecs[d] = ev, V
                row["per_delta"][str(d)] = {"eff_dim_90": e90,
                                            "participation_ratio": pr}
            if not ok:
                print(f"  point {k} incomplete", flush=True)
                continue
            kk = row["per_delta"][str(DELTAS[-1])]["eff_dim_90"]
            row["subspace_cos_10_vs_05"] = subspace_cos(vecs[DELTAS[0]],
                                                        vecs[DELTAS[1]], kk)
            row["subspace_cos_05_vs_025"] = subspace_cos(vecs[DELTAS[1]],
                                                         vecs[DELTAS[2]], kk)
            rows.append(row)
            print(f"  point {k}: eff90 " +
                  " ".join(f"{d}->{row['per_delta'][str(d)]['eff_dim_90']}"
                           for d in DELTAS) +
                  f"   subspace cos {row['subspace_cos_05_vs_025']:.3f}",
                  flush=True)

    def agg(key):
        v = [r["per_delta"][str(key)]["eff_dim_90"] for r in rows]
        return {"mean": float(np.mean(v)), "sd": float(np.std(v))} if v else None

    out = {
        "experiment": "E33_pyloric_fd_convergence",
        "question": "does the rich-observable effective dimension survive a "
                    "finite-difference convergence test?",
        "n_points": len(rows), "deltas": DELTAS,
        "eff_dim_90_by_delta": {str(d): agg(d) for d in DELTAS},
        "subspace_cos_05_vs_025_mean": float(np.mean(
            [r["subspace_cos_05_vs_025"] for r in rows])) if rows else None,
        "subspace_cos_10_vs_05_mean": float(np.mean(
            [r["subspace_cos_10_vs_05"] for r in rows])) if rows else None,
        "per_point": rows,
        "note": "a stable effective dimension and a subspace cosine near 1 mean the "
                "directions are real curvature; a dimension that grows as the step "
                "shrinks and a cosine well below 1 mean finite-difference noise",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open("/root/autodl-tmp/E33_stg_noise.json", "w"), indent=2)
    print("\n=== E-33 RESULT ===", flush=True)
    for d in DELTAS:
        a = out["eff_dim_90_by_delta"][str(d)]
        if a:
            print(f"  delta {d:<6} eff-dim90 {a['mean']:.2f} +-{a['sd']:.2f} of 31",
                  flush=True)
    print(f"  leading-subspace cosine 0.05 vs 0.025: "
          f"{out['subspace_cos_05_vs_025_mean']}", flush=True)
    print("E33_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
