"""Fixed-theta* multistart recovery on BAAIWorm (5d weight-scale, closed-loop NEURON).

Design mirrors _mw_multistart_par.py (modWorm N=30/50 validated) but the forward
model is the GENUINE BAAIWorm NEURON sim, reused verbatim from _genuine_baai_3mode
(scale_weights / run_traj / loss_sim / SPSA / behaviour_closeness_sim) so the
rollout/observables/optimiser are bit-identical to the three-mode genuine arm.

  theta*  = ones (the sim's own theta*, i.e. the SOTA base_w behaviour).
  N inits = random scale=1.0 in multiplicative weight-scale space:
              theta0 ~ exp(U(-ln3, ln3))  (same family as genuine THETA0).
  recover = SPSA (genuine spsa, loss_sim = TrajectoryLoss vs sim-self theta*).

Train/test split: SPSA fits on TRAIN food cond; behaviour closeness (d_init,
d_rec) measured on a DISJOINT TEST food cond. Pure-trajectory only.

Stats: success rate + bootstrap CI95 + Wilcoxon (d_init>d_rec) + manifold
dispersion (param std / behav std) + PCA participation dim.

NEURON ~150-200s/rollout; trials sharded one-per-worker across a Pool that
leaves the running genuine three-mode job (1 core) headroom.
"""
import os, sys, json, time, argparse
import numpy as np
from multiprocessing import Pool
from scipy.stats import ttest_1samp, wilcoxon

sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")
sys.path.insert(0, "/root/BAAIWorm-main/recovery/scripts")
sys.path.insert(0, "/root")

# Reuse the GENUINE three-mode forward model / optimiser verbatim.
import _genuine_baai_3mode as G

OUTDIR = "/root/autodl-tmp/genuine_arch"
os.makedirs(OUTDIR, exist_ok=True)
THETA_STAR = np.ones(5)            # sim's own theta* (base_w @ ones)
TRAIN_FOODS = [50]
TEST_FOODS = [52]
N_ITER = 4                          # SPSA iters per trial (NEURON-lean)


def bootstrap_ci(vals, n_boot=10000, alpha=0.05, seed=7):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(vals) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(vals, size=len(vals), replace=True).mean())
             for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# ---------- per-trial worker ----------
_W = {}

def _init_worker():
    # Each worker loads SOTA base weights + builds its own cond caches once.
    _W["base_w"] = G.load_pickle(G.SOTA)["recovered_weights"]
    _W["train_cache"] = G.build_cond_cache(TRAIN_FOODS)
    _W["test_cache"] = G.build_cond_cache(TEST_FOODS)


def _run_trial(args):
    i, iseed = args
    base_w = _W["base_w"]; train_cache = _W["train_cache"]; test_cache = _W["test_cache"]
    rng = np.random.default_rng(iseed)
    theta0 = np.exp(rng.uniform(-np.log(3), np.log(3), size=5))
    relerr_init = float(np.mean(np.abs(theta0 - THETA_STAR)))
    t0 = time.time()
    # initial behaviour distance to sim-self theta* on TEST conds
    d_init = G.behaviour_closeness_sim(base_w, theta0, test_cache)
    # recover via genuine SPSA on TRAIN (loss_sim = TrajectoryLoss vs sim-self)
    th, bL, L0, hist = G.spsa(
        theta0, lambda t: G.loss_sim(base_w, t, train_cache),
        N_ITER, np.random.default_rng(iseed + 777), f"ms{i}")
    relerr_rec = float(np.mean(np.abs(th - THETA_STAR)))
    d_rec = G.behaviour_closeness_sim(base_w, th, test_cache)
    impr = (d_init - d_rec) / d_init * 100.0 if (d_init and np.isfinite(d_init) and d_init > 0) else np.nan
    return {"idx": i, "init_seed": int(iseed),
            "theta0": theta0.tolist(), "theta_rec": th.tolist(),
            "relerr_init": relerr_init, "relerr_rec": relerr_rec,
            "d_init": d_init, "d_rec": d_rec, "improvement_pct": impr,
            "train_loss_init": L0, "train_loss_best": bL,
            "wall_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="/root/_baai_multistart_N10.json")
    args = ap.parse_args()

    INIT_SEEDS = list(range(3000, 3000 + args.N))   # distinct seed band
    print(f"[baai-multistart] N={args.N} workers={args.workers} n_iter={N_ITER} "
          f"theta*=ones train={TRAIN_FOODS} test={TEST_FOODS}", flush=True)
    t0 = time.time()
    tasks = list(enumerate(INIT_SEEDS))
    results = []
    with Pool(processes=args.workers, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(_run_trial, tasks):
            results.append(r)
            print(f"[trial {r['idx']+1:02d}/{args.N}] d_init={r['d_init']:.4f} "
                  f"d_rec={r['d_rec']:.4f} impr={r['improvement_pct']:5.1f}% "
                  f"relerr {r['relerr_init']:.3f}->{r['relerr_rec']:.3f} "
                  f"L {r['train_loss_init']:.4f}->{r['train_loss_best']:.4f} "
                  f"({r['wall_sec']:.0f}s)", flush=True)
            # incremental dump so partial results survive interruption
            _dump(args, results, t0, partial=True)

    _dump(args, results, t0, partial=False)


def _dump(args, results, t0, partial):
    trials = sorted(results, key=lambda t: t["idx"])
    d_recs = np.array([t["d_rec"] for t in trials], float)
    d_inits = np.array([t["d_init"] for t in trials], float)
    relerr_recs = np.array([t["relerr_rec"] for t in trials], float)
    relerr_inits = np.array([t["relerr_init"] for t in trials], float)
    fin = np.isfinite(d_recs) & np.isfinite(d_inits)
    summary = {"sim": "BAAIWorm_5d", "method": "SPSA", "N_requested": args.N,
               "N_done": len(trials), "partial": partial, "n_iter": N_ITER,
               "theta_star": THETA_STAR.tolist(),
               "init": "random scale=1.0 theta=exp(U(-ln3,ln3))",
               "train_foods": TRAIN_FOODS, "test_foods": TEST_FOODS,
               "loss": "TrajectoryLoss vs sim-self theta* (pure-trajectory)",
               "d_rec_mean": float(np.nanmean(d_recs)) if fin.any() else None,
               "d_rec_std": float(np.nanstd(d_recs)) if fin.any() else None,
               "d_rec_ci95": bootstrap_ci(d_recs[fin]) if fin.any() else [None, None],
               "d_init_mean": float(np.nanmean(d_inits)) if fin.any() else None,
               "d_init_std": float(np.nanstd(d_inits)) if fin.any() else None,
               "d_init_ci95": bootstrap_ci(d_inits[fin]) if fin.any() else [None, None],
               "relerr_rec_mean": float(np.nanmean(relerr_recs)),
               "relerr_init_mean": float(np.nanmean(relerr_inits)),
               "elapsed_sec": time.time() - t0, "trials": trials}
    if fin.sum() >= 2:
        thr = 0.20 * float(np.median(d_inits[fin]))
        success = d_recs[fin] < thr
        summary["success_threshold"] = thr
        summary["success_rate"] = float(success.mean())
        summary["n_success"] = int(success.sum())
        summary["n_better_than_init"] = int((d_recs[fin] < d_inits[fin]).sum())
        diffs = d_inits[fin] - d_recs[fin]
        try:
            ts, tp = ttest_1samp(diffs, 0.0)
            summary["ttest_t"] = float(ts)
            summary["ttest_p_onesided"] = float(tp/2 if ts > 0 else 1 - tp/2)
        except Exception:
            pass
        try:
            ws, wp = wilcoxon(d_inits[fin], d_recs[fin], alternative="greater")
            summary["wilcoxon_stat"] = float(ws); summary["wilcoxon_p_onesided"] = float(wp)
        except Exception:
            summary["wilcoxon_stat"] = None; summary["wilcoxon_p_onesided"] = None
        # manifold dispersion: params scatter vs behaviour clusters
        thetas = np.array([t["theta_rec"] for t in trials])[fin]
        theta_disp = float(np.mean(np.std(thetas / THETA_STAR, axis=0)))
        behav_disp = float(np.std(d_recs[fin]))
        summary["theta_dispersion_normstd"] = theta_disp
        summary["theta_dispersion_perdim"] = np.std(thetas / THETA_STAR, axis=0).tolist()
        summary["behav_dispersion_std"] = behav_disp
        summary["manifold_ratio_param_over_behav"] = theta_disp / (behav_disp + 1e-9)
        try:
            Z = thetas / THETA_STAR; Zc = Z - Z.mean(0)
            sv = np.linalg.svd(Zc, compute_uv=False)
            ev = (sv ** 2) / max(len(Z) - 1, 1); ev = ev / ev.sum()
            summary["manifold_pca_eff_dim_participation"] = float(ev.sum()**2 / np.sum(ev**2))
            summary["manifold_pca_var_ratio"] = ev.tolist()
        except Exception:
            pass
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    if not partial:
        sr = summary.get("success_rate")
        print(f"\n[DONE] BAAIWorm multistart N_done={len(trials)} "
              f"d_rec={summary['d_rec_mean']} d_init={summary['d_init_mean']} "
              f"success={sr} wilcoxon_p={summary.get('wilcoxon_p_onesided')} "
              f"manifold_ratio={summary.get('manifold_ratio_param_over_behav')} "
              f"({summary['elapsed_sec']/60:.1f} min) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
