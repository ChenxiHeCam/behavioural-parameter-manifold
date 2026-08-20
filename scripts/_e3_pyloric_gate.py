"""E-3: complete the pyloric gating sweep and probe every valid point.

Three defects are addressed at once.

(i)  The deposited gating run stopped early (`done: false`, 184 valid) and only
     eight points were probed; the reported effective dimension 2.13/31 is the
     mean of those eight, with per-point values spanning 1.10-3.87 and no
     dispersion reported anywhere.
(ii) A second analysis of the same population in the deposit
     (E1_pyloric_manifold.json, 2365 points) reports 23/31 -- high-dimensional --
     and is cited nowhere. The two must be reconciled.
(iii) The same observable rank cap that inflated the FlyGym result applies here:
     15 summary statistics cannot resolve more than 15 of 31 directions.

So each valid point is probed twice, coarse (15 summary statistics) and rich
(subsampled voltage traces, thousands of observables), and the full distribution
is reported rather than a mean over eight.
"""
import os
for _v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'   # 172 processes each opening a BLAS pool oversubscribes the box
import json, os, sys, time
import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# the pyloric simulator is a Cython extension and is not fork-safe: forked pools
# die mid-map. spawn gives each worker a clean interpreter.
CTX = mp.get_context("spawn")

N_DRAW = 40000        # completes the sweep rather than stopping early
DELTA = 0.05
SUB = 200             # voltage subsampling stride
K_MAX = 60            # valid points to probe with the full Hessian
SEED = 0


def _sim(theta_vals):
    from pyloric import simulate, summary_stats, create_prior
    import numpy as np
    pr = create_prior()
    s = pr.sample((1,)).astype(float)   # the sampled frame is float32; a perturbed
    s.loc[0] = np.asarray(theta_vals, dtype=float)   # value need not be representable
    out = simulate(s.loc[0])
    v = np.asarray(out["voltage"], dtype=np.float32)
    ss = np.asarray(summary_stats(out), dtype=float).ravel()
    return ss, v[:, ::SUB]


def draw_and_gate(i):
    """Draw one prior sample, simulate, and return it if the rhythm is valid."""
    import numpy as np
    from pyloric import create_prior
    pr = create_prior()
    np.random.seed((SEED + i) % (1 << 31))
    s = pr.sample((1,))
    th = np.asarray(s.loc[0], dtype=float)
    try:
        ss, _ = _sim(th)
    except Exception:
        return None
    if not np.all(np.isfinite(ss)):
        return None
    period = ss[0]
    if not (0.5 <= period <= 2.0 or 500 <= period <= 2000):
        return None
    return th.tolist(), ss.tolist()


def probe(job):
    """One finite-difference leg of one point's Hessian."""
    import numpy as np
    k, th, idx, sign = job
    th = np.array(th, dtype=float)
    if idx >= 0:
        th[idx] *= (1.0 + sign * DELTA)
    try:
        ss, v = _sim(th)
    except Exception:
        return k, idx, sign, None, None
    return k, idx, sign, ss, v.astype(np.float32)


def eff_dim(ev):
    ev = np.maximum(np.real(ev), 0.0)
    ev = ev[ev > 0]
    if ev.size == 0:
        return 0, 0, 0.0
    s = np.sort(ev)[::-1]
    cs = np.cumsum(s) / s.sum()
    return (int(np.searchsorted(cs, 0.90) + 1), int(np.searchsorted(cs, 0.99) + 1),
            float(s.sum() ** 2 / (s ** 2).sum()))


def main():
    t0 = time.time()
    nw = 40
    print(f"stage 1: gating {N_DRAW} prior draws on {nw} workers", flush=True)

    valid = []
    done = 0
    block = 4000
    for start in range(0, N_DRAW, block):
        idxs = range(start, min(start + block, N_DRAW))
        try:
            with ProcessPoolExecutor(max_workers=nw, mp_context=CTX, max_tasks_per_child=50) as ex:
                for r in ex.map(draw_and_gate, idxs, chunksize=10):
                    done += 1
                    if r is not None:
                        valid.append(r)
        except Exception as e:
            print(f"  block {start} aborted ({type(e).__name__}), continuing", flush=True)
        print(f"  {done}/{N_DRAW} drawn, {len(valid)} valid, "
              f"{time.time()-t0:.0f}s", flush=True)
    rate = len(valid) / max(done, 1)
    print(f"stage 1 done: {len(valid)} valid of {N_DRAW} ({100*rate:.2f}%), "
          f"{time.time()-t0:.0f}s", flush=True)

    json.dump([v[0] for v in valid],
              open("/root/autodl-tmp/E3_valid_points.json", "w"))
    K = min(K_MAX, len(valid))
    pts = [valid[i][0] for i in np.linspace(0, len(valid) - 1, K).astype(int)]

    jobs = []
    for k, th in enumerate(pts):
        jobs.append((k, th, -1, 0))
        for i in range(31):
            jobs.append((k, th, i, +1))
            jobs.append((k, th, i, -1))
    print(f"stage 2: {len(jobs)} simulations for {K} Hessians", flush=True)

    store = {}
    nb = 2000
    for start in range(0, len(jobs), nb):
        chunk = jobs[start:start + nb]
        try:
            with ProcessPoolExecutor(max_workers=nw, mp_context=CTX, max_tasks_per_child=50) as ex:
                for k, idx, sign, ss, v in ex.map(probe, chunk, chunksize=2):
                    store[(k, idx, sign)] = (ss, v)
        except Exception as e:
            print(f"  probe block {start} aborted ({type(e).__name__})", flush=True)
        print(f"  {min(start+nb,len(jobs))}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    Tm = 0
    coarse, rich = [], []
    drop = {"no_baseline": 0, "missing_leg": 0, "ok": 0}
    for k in range(K):
        b = store.get((k, -1, 0))
        if b is None or b[1] is None:
            drop["no_baseline"] += 1
            continue
        bss, bv = b
        T = bv.shape[1]
        Jc, Jr, ok = [], [], True
        for i in range(31):
            p, m = store.get((k, i, +1)), store.get((k, i, -1))
            if p is None or m is None or p[1] is None or m[1] is None:
                ok = False
                break
            Tm = min(T, p[1].shape[1], m[1].shape[1])
            Jc.append((p[0] - m[0]) / (2 * DELTA))
            Jr.append(((p[1][:, :Tm] - m[1][:, :Tm]) / (2 * DELTA)).astype(float).ravel())
        if not ok:
            drop["missing_leg"] += 1
            continue
        drop["ok"] += 1
        Tm = min(len(x) for x in Jr) // 3
        Jc = np.array(Jc).T
        Jr = np.array([x[:3 * Tm] for x in Jr]).T
        wc = 1.0 / np.maximum(np.abs(bss), 1e-8)
        Hc = (Jc * wc[:, None]).T @ (Jc * wc[:, None])
        sv = bv[:, :Tm].std(1).clip(1e-3)
        wr = np.repeat(1.0 / sv, Tm)
        Hr = (Jr * wr[:, None]).T @ (Jr * wr[:, None])
        if np.all(np.isfinite(Hc)):
            coarse.append(eff_dim(np.linalg.eigvalsh(Hc)))
        if np.all(np.isfinite(Hr)):
            rich.append(eff_dim(np.linalg.eigvalsh(Hr)))

    print(f"point accounting: {drop}", flush=True)
    print(f"coarse Hessians usable: {len(coarse)}, rich: {len(rich)}", flush=True)

    def summarise(rows, nobs):
        if not rows:
            return {"n_points": 0, "n_observables": nobs,
                    "note": "no point yielded a finite Hessian at this observable set"}
        a = np.array(rows, dtype=float)
        return {"n_points": len(rows), "n_observables": nobs,
                "eff_dim_90": {"mean": float(a[:, 0].mean()), "sd": float(a[:, 0].std()),
                               "min": float(a[:, 0].min()), "max": float(a[:, 0].max())},
                "eff_dim_99": {"mean": float(a[:, 1].mean()), "sd": float(a[:, 1].std()),
                               "min": float(a[:, 1].min()), "max": float(a[:, 1].max())},
                "participation_ratio": {"mean": float(a[:, 2].mean()),
                                        "sd": float(a[:, 2].std())},
                "per_point_eff_dim_90": [int(x) for x in a[:, 0]]}

    out = {
        "experiment": "E3_pyloric_complete",
        "n_drawn": N_DRAW, "n_valid": len(valid), "valid_rate": rate,
        "gating_completed": True,
        "n_points_probed": len(coarse), "n_params": 31, "delta": DELTA,
        "coarse": summarise(coarse, 15),
        "rich": summarise(rich, int(3 * Tm)),
        "published": {"eff_dim_mean": 2.13, "K": 8, "note": "mean of eight points, "
                      "coarse observables, incomplete gating sweep"},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open("/root/autodl-tmp/E3_pyloric_complete.json", "w"), indent=2)
    print("\n=== E-3 RESULT ===", flush=True)
    print(f"gating: {len(valid)}/{N_DRAW} valid ({100*rate:.2f}%), sweep completed", flush=True)
    c, r = out["coarse"], out["rich"]
    print(f"coarse 15 obs : eff-dim90 {c['eff_dim_90']['mean']:.2f} "
          f"+-{c['eff_dim_90']['sd']:.2f} (range {c['eff_dim_90']['min']:.0f}"
          f"-{c['eff_dim_90']['max']:.0f}) over {c['n_points']} points", flush=True)
    print(f"rich {r['n_observables']} obs : eff-dim90 {r['eff_dim_90']['mean']:.2f} "
          f"+-{r['eff_dim_90']['sd']:.2f} (range {r['eff_dim_90']['min']:.0f}"
          f"-{r['eff_dim_90']['max']:.0f})", flush=True)
    print("E3_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
