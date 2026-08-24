"""AB2d: confirm the summary-statistic proportionality result under a second whitening.

AB2c whitened each summary statistic by its own baseline magnitude, which
inflates any statistic that happens to sit near zero at the operating point. If
the failure of proportionality were an artefact of that choice it would go away
under a whitening by the across-population spread of each statistic, which is
the scale a Gauss-Newton weighting would normally use. Both are computed here
from the same simulations, so the two answers are directly comparable.

Raw statistics are saved so any third whitening can be applied without re-running.
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
OUT = "/root/autodl-tmp/cell/AB2d_stg_whitening.json"
DELTAS = [0.01, 0.05, 0.25]
K = int(os.environ.get("KPT", "12"))
NW = int(os.environ.get("NW", "30"))
_COLS = None


def _init():
    global _COLS
    from pyloric import create_prior
    _COLS = create_prior().sample((1,)).columns


def leg(a):
    import numpy as np, pandas as pd
    from pyloric import simulate, summary_stats
    k, idx, delta, th = a
    th = np.array(th, float)
    if idx >= 0:
        th[idx] *= (1.0 + delta)
    try:
        out = simulate(pd.Series(th, index=_COLS))
        s = np.asarray(summary_stats(out), dtype=float).ravel()
        return (k, idx, delta), (s.tolist() if np.all(np.isfinite(s)) else None)
    except Exception:
        return (k, idx, delta), None


def main():
    t0 = time.time()
    P = np.load(PARAMS)
    pts = P[np.random.default_rng(0).choice(len(P), size=K, replace=False)]
    jobs = [(k, -1, 0.0, pts[k]) for k in range(K)]
    jobs += [(k, i, d, pts[k]) for k in range(K) for i in range(31) for d in DELTAS]
    res = {}
    with ProcessPoolExecutor(max_workers=NW, mp_context=CTX, initializer=_init) as ex:
        for n, (key, val) in enumerate(ex.map(leg, jobs, chunksize=4)):
            res[key] = val
            if (n + 1) % 300 == 0:
                print(f"  {n+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    bases = {k: np.array(res[(k, -1, 0.0)]) for k in range(K) if res.get((k, -1, 0.0))}
    if not bases:
        raise SystemExit("no usable baselines")
    pop_sd = np.std(np.array(list(bases.values())), axis=0)
    pop_sd = np.where(pop_sd > 1e-9, pop_sd, 1.0)

    out = {"baseline": "own magnitude", "population": "across-point sd"}
    spreads = {"baseline": [], "population": []}
    raw = []
    for k, b in bases.items():
        own = np.where(np.abs(b) > 1e-9, np.abs(b), 1.0)
        for i in range(31):
            legs = {d: res.get((k, i, d)) for d in DELTAS}
            if any(v is None for v in legs.values()):
                continue
            rec = {"k": k, "param": i, "baseline_stats": b.tolist()}
            for tag, sd in (("baseline", own), ("population", pop_sd)):
                r = []
                for d in DELTAS:
                    n_ = float(np.linalg.norm((np.array(legs[d]) - b) / sd) / np.sqrt(len(b)))
                    r.append(n_ / d)
                    rec[f"norm_{tag}_{d}"] = n_
                spreads[tag].append(max(r) / min(r) if min(r) > 0 else float("inf"))
            raw.append(rec)

    sat = max(DELTAS) / min(DELTAS)
    result = {"experiment": "AB2d_stg_whitening_check", "deltas": DELTAS,
              "n_pairs": len(raw), "saturated_reference_spread": sat,
              "whitening": out, "per_pair": raw}
    for tag in ("baseline", "population"):
        v = np.array([x for x in spreads[tag] if np.isfinite(x)])
        result[f"ratio_spread_{tag}"] = {
            "median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
            "q75": float(np.percentile(v, 75)), "n": int(len(v)),
            "fraction_proportional": float(np.mean(v < sat ** 0.5))}
        print(f"{tag:11s} median spread {np.median(v):6.2f}  "
              f"q25 {np.percentile(v,25):6.2f}  q75 {np.percentile(v,75):6.2f}  "
              f"proportional {np.mean(v < sat**0.5):.3f}  (saturated = {sat:.0f})")
    json.dump(result, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
