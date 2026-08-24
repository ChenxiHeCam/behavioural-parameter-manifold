"""AB2: the pyloric finite-difference argument, measured rather than asserted.

The paper withdraws the trace-based pyloric dimension on the grounds that a
perturbation shifts the phase of the rhythm, so the trace difference saturates
at the scale of the oscillation instead of shrinking with the step. If that is
right then the response norm is flat in delta, the Jacobian scales as 1/delta,
the normalised spectrum is invariant by construction, and a step-size
convergence test is uninformative rather than passed. That last distinction
matters: E33's own preregistered criterion (stable dimension plus a subspace
cosine near one) reads as "real curvature" on the trace observables, so the
withdrawal has to rest on the scaling argument and the scaling has to be shown.

Two quantities are measured here, on the same points and the same legs:

  1. the whitened response norm against step size, separately for the 6600
     trace observables and for the 15 phase-invariant summary statistics. A
     proportional response gives a straight line through the origin on the
     norm-vs-delta plot; a saturating one gives a plateau.

  2. the Pearson correlation between the baseline and perturbed voltage traces
     at each step, which is the quantity quoted in S1.5 as 0.42 at 0.1% and
     0.06 at 25% and which had no script in the repository.
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
OUT = "/root/autodl-tmp/cell/AB2b_stg_scaling.json"
DELTAS = [0.001, 0.01, 0.05, 0.10, 0.25]
SUB = 400
K = int(os.environ.get("KPT", "24"))
NW = int(os.environ.get("NW", "30"))
SEED = 0
_COLS = None


def _init():
    global _COLS
    from pyloric import create_prior
    _COLS = create_prior().sample((1,)).columns


def _sim(vals):
    import numpy as np, pandas as pd
    from pyloric import simulate, summary_stats
    out = simulate(pd.Series(np.asarray(vals, float), index=_COLS))
    v = np.asarray(out["voltage"], dtype=np.float32)
    ss = np.asarray(summary_stats(out), dtype=float).ravel()
    return ss, v


def leg(a):
    """One leg. idx = -1 is the baseline; otherwise parameter idx scaled by 1+delta."""
    k, idx, delta, th = a
    th = np.array(th, float)
    if idx >= 0:
        th[idx] *= (1.0 + delta)
    try:
        ss, v = _sim(th)
        if not np.all(np.isfinite(ss)):
            return (k, idx, delta), None
        return (k, idx, delta), (ss, v[:, ::SUB].astype(np.float32), v[:, ::10].astype(np.float32))
    except Exception:
        return (k, idx, delta), None


def main():
    t0 = time.time()
    P = np.load(PARAMS)
    rng = np.random.default_rng(SEED)
    pts = P[rng.choice(len(P), size=K, replace=False)]

    # one parameter is perturbed for the scan: index 0 is AB/PD Na, the
    # conductance S1.5 names when it quotes the decorrelation figures
    IDX = 0
    jobs = [(k, -1, 0.0, pts[k]) for k in range(K)]
    jobs += [(k, IDX, d, pts[k]) for k in range(K) for d in DELTAS]

    res = {}
    with ProcessPoolExecutor(max_workers=NW, mp_context=CTX, initializer=_init) as ex:
        for i, (key, val) in enumerate(ex.map(leg, jobs, chunksize=2)):
            res[key] = val
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    rows = []
    for k in range(K):
        b = res.get((k, -1, 0.0))
        if b is None:
            continue
        b_ss, b_sub, b_fine = b
        rec = {"k": k, "steps": []}
        for d in DELTAS:
            p = res.get((k, IDX, d))
            if p is None:
                continue
            p_ss, p_sub, p_fine = p
            # whitened response norm through the summary statistics
            sd = np.abs(b_ss)
            sd = np.where(sd > 1e-9, sd, 1.0)
            n_ss = float(np.linalg.norm((p_ss - b_ss) / sd) / np.sqrt(len(b_ss)))
            # whitened response norm through the subsampled trace
            t = min(b_sub.shape[1], p_sub.shape[1])
            a, c = b_sub[:, :t].astype(float), p_sub[:, :t].astype(float)
            s = a.std(1, keepdims=True).clip(1e-3)
            n_tr = float(np.linalg.norm((c - a) / s) / np.sqrt(a.size))
            # Pearson correlation on the fine trace, per cell then averaged
            t2 = min(b_fine.shape[1], p_fine.shape[1])
            rs = []
            for ci in range(b_fine.shape[0]):
                x, y = b_fine[ci, :t2].astype(float), p_fine[ci, :t2].astype(float)
                if x.std() > 1e-9 and y.std() > 1e-9:
                    rs.append(float(np.corrcoef(x, y)[0, 1]))
            rec["steps"].append({"delta": d, "norm_summary_stats": n_ss,
                                 "norm_trace": n_tr,
                                 "trace_pearson_r": float(np.mean(rs)) if rs else None})
        if rec["steps"]:
            rows.append(rec)

    def agg(field):
        out = []
        for d in DELTAS:
            v = [s[field] for r in rows for s in r["steps"]
                 if s["delta"] == d and s[field] is not None]
            if v:
                out.append({"delta": d, "mean": float(np.mean(v)),
                            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                            "n": len(v),
                            "mean_over_delta": float(np.mean(v)) / d})
        return out

    result = {
        "experiment": "AB2b_stg_response_scaling",
        "question": "does the trace response saturate in the step while the summary statistics stay proportional?",
        "n_points_requested": K, "n_points_usable": len(rows),
        "perturbed_parameter_index": IDX, "voltage_stride": SUB,
        "deltas": DELTAS,
        "summary_stat_response": agg("norm_summary_stats"),
        "trace_response": agg("norm_trace"),
        "trace_pearson_r": agg("trace_pearson_r"),
        "per_point": rows,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(result, open(OUT, "w"), indent=1)
    print("saved ->", OUT, flush=True)
    print(f"{'delta':>7} {'|dObs| stats':>13} {'ratio/delta':>12} "
          f"{'|dObs| trace':>13} {'ratio/delta':>12} {'trace r':>9}")
    for a, b, c in zip(result["summary_stat_response"], result["trace_response"],
                       result["trace_pearson_r"]):
        print(f"{a['delta']:7.3f} {a['mean']:13.5g} {a['mean_over_delta']:12.4g} "
              f"{b['mean']:13.5g} {b['mean_over_delta']:12.4g} {c['mean']:9.3f}")


if __name__ == "__main__":
    main()
