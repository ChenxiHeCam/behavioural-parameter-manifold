"""AB2c: is the pyloric summary-statistic response proportional to the step?

AB2b showed, for one conductance, that the trace response is flat in the step
(so the trace Jacobian scales as 1/delta and the normalised spectrum is
invariant by construction) but also that the 15 summary statistics did not
respond proportionally either. One conductance is not enough to decide that:
AB/PD Na is the conductance the pyloric literature calls compensated, so an
outsized response to a 0.1% change would be surprising for a different reason.

This runs the same scan over all 31 conductances, so the question becomes a
distribution rather than an anecdote: for what fraction of parameters is the
whitened summary-statistic response proportional to the step, and does the
coarse Gauss-Newton curvature that carries the reported dimension rest on
proportional legs or on discontinuous ones?
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import json, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

CTX = mp.get_context("spawn")
PARAMS = "/root/autodl-tmp/cell/e1_valid_params.npy"
OUT = "/root/autodl-tmp/cell/AB2c_stg_allparam.json"
DELTAS = [0.01, 0.05, 0.25]
K = int(os.environ.get("KPT", "12"))
NW = int(os.environ.get("NW", "30"))
SEED = 0
_COLS = None


def _init():
    global _COLS
    from pyloric import create_prior
    _COLS = create_prior().sample((1,)).columns


def _ss(vals):
    import numpy as np, pandas as pd
    from pyloric import simulate, summary_stats
    out = simulate(pd.Series(np.asarray(vals, float), index=_COLS))
    return np.asarray(summary_stats(out), dtype=float).ravel()


def leg(a):
    k, idx, delta, th = a
    th = np.array(th, float)
    if idx >= 0:
        th[idx] *= (1.0 + delta)
    try:
        s = _ss(th)
        return (k, idx, delta), (s if np.all(np.isfinite(s)) else None)
    except Exception:
        return (k, idx, delta), None


def main():
    t0 = time.time()
    P = np.load(PARAMS)
    rng = np.random.default_rng(SEED)
    pts = P[rng.choice(len(P), size=K, replace=False)]

    jobs = [(k, -1, 0.0, pts[k]) for k in range(K)]
    jobs += [(k, i, d, pts[k]) for k in range(K) for i in range(31) for d in DELTAS]
    print(f"{len(jobs)} simulations", flush=True)

    res = {}
    with ProcessPoolExecutor(max_workers=NW, mp_context=CTX, initializer=_init) as ex:
        for n, (key, val) in enumerate(ex.map(leg, jobs, chunksize=4)):
            res[key] = val
            if (n + 1) % 300 == 0:
                print(f"  {n+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    per_param, prop_flags = [], []
    for k in range(K):
        b = res.get((k, -1, 0.0))
        if b is None:
            continue
        sd = np.where(np.abs(b) > 1e-9, np.abs(b), 1.0)
        for i in range(31):
            norms = {}
            for d in DELTAS:
                p = res.get((k, i, d))
                if p is None:
                    continue
                norms[d] = float(np.linalg.norm((p - b) / sd) / np.sqrt(len(b)))
            if len(norms) < len(DELTAS):
                continue
            # proportional means the norm/step ratio is roughly constant; a
            # saturating response makes it fall like 1/delta
            ratios = [norms[d] / d for d in DELTAS]
            lo, hi = min(ratios), max(ratios)
            spread = hi / lo if lo > 0 else float("inf")
            step_span = max(DELTAS) / min(DELTAS)          # 25
            per_param.append({"k": k, "param": i,
                              "norms": {str(d): norms[d] for d in DELTAS},
                              "ratio_spread": spread})
            # proportional if the ratio varies by less than a quarter of what a
            # fully saturated response would produce
            prop_flags.append(spread < step_span ** 0.5)

    spreads = np.array([r["ratio_spread"] for r in per_param])
    frac = float(np.mean(prop_flags)) if len(prop_flags) else None
    result = {
        "experiment": "AB2c_stg_summary_stat_proportionality",
        "question": "do the 15 phase-invariant summary statistics respond proportionally to the step?",
        "deltas": DELTAS, "n_points": K, "n_param_point_pairs": len(per_param),
        "saturated_reference_spread": max(DELTAS) / min(DELTAS),
        "proportionality_threshold": (max(DELTAS) / min(DELTAS)) ** 0.5,
        "fraction_proportional": frac,
        "ratio_spread_median": float(np.median(spreads)) if len(spreads) else None,
        "ratio_spread_q25": float(np.percentile(spreads, 25)) if len(spreads) else None,
        "ratio_spread_q75": float(np.percentile(spreads, 75)) if len(spreads) else None,
        "per_param_point": per_param,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(result, open(OUT, "w"), indent=1)
    print("saved ->", OUT)
    print(f"pairs={len(per_param)}  fraction proportional={frac:.3f}")
    print(f"ratio spread: median {np.median(spreads):.2f}  "
          f"q25 {np.percentile(spreads,25):.2f}  q75 {np.percentile(spreads,75):.2f}  "
          f"(fully saturated would give {max(DELTAS)/min(DELTAS):.0f})")


if __name__ == "__main__":
    main()
