"""Parallel fixed-theta* multistart recovery on modWorm (30-dim), N-scalable.

Identical forward model / observables / optimisers / seeds as _mw_multistart.py
(N=30 validated), but the per-trial loop is sharded across a multiprocessing
pool so N=50 finishes within a few minutes of wall time while leaving cores
free for the BAAIWorm three-mode job.

Seed scheme is a strict SUPERSET of the N=30 run:
  INIT_SEEDS = range(2000, 2000+N)  -> N=50 reuses 2000..2029 then adds 2030..2049
so the first 30 trials are bit-identical to the prior N=30 result.

Pure-trajectory observables only. Deterministic. Train/test rollout-seed split.
Manifold dispersion (Exp 4) computed in summary.
"""
import os, sys, time, json, argparse
import numpy as np
from multiprocessing import Pool
from scipy.stats import wasserstein_distance, ttest_1samp, wilcoxon

N_SEG = 10; N_STEPS = 200; DT = 0.005

THETA_GT = np.concatenate([
    np.full(10, 0.50), np.full(10, 1.20),
    np.array([1.00, 0.10, 0.020, -70.0, 0.05]),
    np.array([0.80, 0.78, 0.75, 0.72, 0.70]),
])
assert THETA_GT.shape == (30,)
NAMES = ([f"gap_{i}" for i in range(10)] + [f"syn_{i}" for i in range(10)] +
         ["Cm", "Gleak", "tau", "Eleak", "g_in"] + [f"mus_{i}" for i in range(5)])


def rollout(theta, seed=0):
    gap = theta[0:10]; syn = theta[10:20]
    Cm, Gleak, tau, Eleak, g_in = theta[20:25]
    mus = np.repeat(theta[25:30], 2)[:N_SEG]
    rng = np.random.RandomState(seed)
    V = np.full((N_STEPS, N_SEG), Eleak, dtype=np.float64)
    Iext = rng.uniform(-0.5, 0.5, size=(N_STEPS, N_SEG))
    Iext += 0.3 * np.sin(2 * np.pi * np.arange(N_STEPS)[:, None] * 0.02
                         + np.arange(N_SEG)[None, :] * 0.4)
    for t in range(1, N_STEPS):
        prev = V[t - 1]
        lap = np.zeros_like(prev)
        lap[1:-1] = prev[:-2] - 2 * prev[1:-1] + prev[2:]
        lap[0] = prev[1] - prev[0]; lap[-1] = prev[-2] - prev[-1]
        syn_in = syn * np.tanh(0.05 * (np.roll(prev, 1) + np.roll(prev, -1)))
        dV = (-Gleak * (prev - Eleak) + gap * lap + syn_in
              + mus * Iext[t] + g_in * np.tanh(prev - Eleak)) / (Cm * tau * 100.0)
        V[t] = prev + DT * dV
        if not np.all(np.isfinite(V[t])):
            return None
    bend = np.cumsum(np.tanh(0.02 * (V - Eleak)), axis=0) * 0.01
    return bend


OBS_NAMES = ["bend_amp", "bend_mean", "head_freq", "propagation",
             "speed_proxy", "turn_rate_proxy"]


def observables(bend):
    mid = bend.mean(axis=1); amp = bend.std(axis=0)
    head = bend[:, 0]; tail = bend[:, -1]
    dbend = np.diff(bend, axis=0)
    fft = np.abs(np.fft.rfft(head - head.mean()))
    head_freq = float(fft[1:].argmax() + 1) if len(fft) > 1 else 0.0
    if head.std() > 1e-9 and tail.std() > 1e-9:
        prop = float(np.corrcoef(head, tail)[0, 1])
    else:
        prop = 0.0
    return np.array([float(amp.mean()), float(bend.mean()), head_freq, prop,
                     float(np.abs(dbend).mean()), float(np.abs(np.diff(mid)).mean())])


def collect_obs(theta, seeds):
    out = []
    for s in seeds:
        b = rollout(theta, seed=s)
        if b is None:
            continue
        out.append(observables(b))
    return np.array(out) if out else None


def behav_dist(obs_a, obs_target, scale_mad):
    if obs_a is None or len(obs_a) == 0:
        return np.nan
    d = 0.0
    for j in range(len(OBS_NAMES)):
        a = obs_a[:, j] / scale_mad[j]; t = obs_target[:, j] / scale_mad[j]
        d += wasserstein_distance(a, t)
    return float(d / len(OBS_NAMES))


def loss_mse(theta, target_trajs, seeds):
    vals = []
    for s, tg in zip(seeds, target_trajs):
        tr = rollout(theta, seed=s)
        if tr is None or not np.all(np.isfinite(tr)):
            continue
        vals.append(float(np.mean((tr - tg) ** 2)))
    return float(np.mean(vals)) if vals else 1.0


def spsa(theta0, target_trajs, train_seeds, n_iter, seed):
    rng = np.random.default_rng(seed); K = len(theta0)
    scale_vec = np.abs(THETA_GT); a, c = 0.20, 0.08
    x = (theta0 - THETA_GT) / scale_vec; x_best = x.copy()
    L_best = loss_mse(THETA_GT + x * scale_vec, target_trajs, train_seeds)
    for it in range(n_iter):
        ck = c / (it + 1) ** 0.101; ak = a / (it + 1) ** 0.602
        bk = rng.choice([-1.0, 1.0], size=K)
        xp = x + ck * bk; xm = x - ck * bk
        Lp = loss_mse(THETA_GT + xp * scale_vec, target_trajs, train_seeds)
        Lm = loss_mse(THETA_GT + xm * scale_vec, target_trajs, train_seeds)
        if not (np.isfinite(Lp) and np.isfinite(Lm)):
            continue
        ghat = (Lp - Lm) / (2 * ck * max(L_best, 1e-9)) * bk
        step = np.clip(ak * ghat, -0.1, 0.1)
        x = x - step
        Lc = loss_mse(THETA_GT + x * scale_vec, target_trajs, train_seeds)
        if np.isfinite(Lc) and Lc < L_best:
            L_best = Lc; x_best = x.copy()
    return THETA_GT + x_best * scale_vec


def cma(theta0, target_trajs, train_seeds, n_iter, seed, popsize=16):
    import cma as cmalib
    scale_vec = np.abs(THETA_GT)
    x0 = (theta0 - THETA_GT) / scale_vec
    es = cmalib.CMAEvolutionStrategy(x0.tolist(), 0.05, {
        "popsize": popsize, "maxiter": n_iter, "verbose": -9,
        "CMA_diagonal": True, "seed": int(seed) % (2**31 - 1)})
    gen = 0
    while not es.stop() and gen < n_iter:
        xs = es.ask()
        losses = [loss_mse(THETA_GT + np.array(xi) * scale_vec, target_trajs, train_seeds)
                  for xi in xs]
        es.tell(xs, losses); gen += 1
    xbest = np.array(es.result.xbest) if es.result.xbest is not None else np.array(es.mean)
    return THETA_GT + xbest * scale_vec


def bootstrap_ci(vals, n_boot=10000, alpha=0.05, seed=7):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(vals) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(vals, size=len(vals), replace=True).mean())
             for _ in range(n_boot)]
    return [float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2)))]


# ---- per-trial worker (module-level for picklability) ----
_G = {}

def _init_worker(method, n_iter, init_scale, train_seeds, test_seeds,
                 target_trajs_train, obs_target_test, scale_mad):
    _G.update(dict(method=method, n_iter=n_iter, init_scale=init_scale,
                   train_seeds=train_seeds, test_seeds=test_seeds,
                   target_trajs_train=target_trajs_train,
                   obs_target_test=np.array(obs_target_test),
                   scale_mad=np.array(scale_mad)))


def _run_trial(args):
    i, iseed = args
    g = _G
    rng = np.random.RandomState(iseed)
    theta_init = THETA_GT * (1.0 + g["init_scale"] * rng.randn(len(THETA_GT)))
    obs_init_test = collect_obs(theta_init, g["test_seeds"])
    d_init = behav_dist(obs_init_test, g["obs_target_test"], g["scale_mad"])
    relerr_init = float(np.mean(np.abs((theta_init - THETA_GT) / THETA_GT)))
    ti = time.time()
    if g["method"] == "spsa":
        theta_rec = spsa(theta_init, g["target_trajs_train"], g["train_seeds"], g["n_iter"], iseed)
    else:
        theta_rec = cma(theta_init, g["target_trajs_train"], g["train_seeds"], g["n_iter"], iseed)
    obs_rec_test = collect_obs(theta_rec, g["test_seeds"])
    d_rec = behav_dist(obs_rec_test, g["obs_target_test"], g["scale_mad"])
    relerr_rec = float(np.mean(np.abs((theta_rec - THETA_GT) / THETA_GT)))
    improvement = (d_init - d_rec) / d_init * 100.0 if d_init > 0 else np.nan
    return {
        "idx": i, "init_seed": int(iseed),
        "d_init": d_init, "d_rec": d_rec, "improvement_pct": improvement,
        "relerr_init": relerr_init, "relerr_rec": relerr_rec,
        "theta_rec": theta_rec.tolist(), "wall_sec": time.time() - ti,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--n_iter", type=int, default=50)
    ap.add_argument("--method", choices=["spsa", "cma"], default="cma")
    ap.add_argument("--init_scale", type=float, default=0.30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/root/_mw_multistart_cma_N50.json")
    args = ap.parse_args()

    TRAIN_SEEDS = list(range(0, 8)); TEST_SEEDS = list(range(100, 116))
    target_trajs_train = [rollout(THETA_GT, seed=s) for s in TRAIN_SEEDS]
    obs_target_test = collect_obs(THETA_GT, TEST_SEEDS)
    scale_mad = np.array([
        (np.median(np.abs(obs_target_test[:, j] - np.median(obs_target_test[:, j]))) + 1e-9)
        for j in range(len(OBS_NAMES))])
    INIT_SEEDS = list(range(2000, 2000 + args.N))

    print(f"[mw-multistart-par] method={args.method} N={args.N} n_iter={args.n_iter} "
          f"init_scale={args.init_scale} workers={args.workers}", flush=True)
    t0 = time.time()
    tasks = list(enumerate(INIT_SEEDS))
    init_args = (args.method, args.n_iter, args.init_scale, TRAIN_SEEDS, TEST_SEEDS,
                 target_trajs_train, obs_target_test.tolist(), scale_mad.tolist())
    with Pool(processes=args.workers, initializer=_init_worker, initargs=init_args) as pool:
        results = []
        for r in pool.imap_unordered(_run_trial, tasks):
            results.append(r)
            print(f"[trial {r['idx']+1:02d}/{args.N}] d_init={r['d_init']:.4f} "
                  f"d_rec={r['d_rec']:.4f} impr={r['improvement_pct']:5.1f}% "
                  f"relerr {r['relerr_init']:.3f}->{r['relerr_rec']:.3f} "
                  f"({r['wall_sec']:.1f}s)", flush=True)
    trials = sorted(results, key=lambda t: t["idx"])
    for t in trials:
        t.pop("idx", None)

    d_recs = np.array([t["d_rec"] for t in trials], float)
    d_inits = np.array([t["d_init"] for t in trials], float)
    relerr_recs = np.array([t["relerr_rec"] for t in trials], float)
    relerr_inits = np.array([t["relerr_init"] for t in trials], float)

    thr = 0.20 * float(np.median(d_inits))
    success = d_recs < thr
    success_rate = float(success.mean())
    diffs = d_inits - d_recs
    try:
        t_stat, t_p = ttest_1samp(diffs, 0.0)
        t_p_one = t_p / 2 if t_stat > 0 else 1 - t_p / 2
    except Exception:
        t_stat, t_p_one = None, None
    try:
        w_stat, w_p = wilcoxon(d_inits, d_recs, alternative="greater")
    except Exception:
        w_stat, w_p = None, None
    n_better = int((d_recs < d_inits).sum())

    thetas = np.array([t["theta_rec"] for t in trials])
    theta_disp = float(np.mean(np.std(thetas / np.abs(THETA_GT), axis=0)))
    # per-dimension dispersion (manifold breadth profile)
    theta_disp_perdim = np.std(thetas / np.abs(THETA_GT), axis=0).tolist()
    behav_disp = float(np.std(d_recs))
    mean_d_rec = float(np.nanmean(d_recs))
    manifold_ratio = theta_disp / (mean_d_rec + 1e-9)
    # PCA of recovered theta (normalised) -> participation / effective dim
    Z = thetas / np.abs(THETA_GT)
    Zc = Z - Z.mean(0)
    try:
        svals = np.linalg.svd(Zc, compute_uv=False)
        ev = (svals ** 2) / max((len(Z) - 1), 1)
        ev = ev / ev.sum()
        eff_dim = float((ev.sum() ** 2) / (np.sum(ev ** 2)))  # participation ratio
        pca_var_ratio = ev.tolist()
    except Exception:
        eff_dim, pca_var_ratio = None, None

    summary = {
        "sim": "modWorm_30d", "method": args.method.upper(), "N": args.N,
        "n_iter": args.n_iter, "init_scale": args.init_scale,
        "train_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS,
        "obs_names": OBS_NAMES, "theta_gt": THETA_GT.tolist(),
        "d_rec_mean": float(np.nanmean(d_recs)), "d_rec_std": float(np.nanstd(d_recs)),
        "d_rec_ci95": bootstrap_ci(d_recs),
        "d_init_mean": float(np.nanmean(d_inits)), "d_init_std": float(np.nanstd(d_inits)),
        "d_init_ci95": bootstrap_ci(d_inits),
        "relerr_rec_mean": float(np.nanmean(relerr_recs)),
        "relerr_rec_std": float(np.nanstd(relerr_recs)),
        "relerr_rec_ci95": bootstrap_ci(relerr_recs),
        "relerr_init_mean": float(np.nanmean(relerr_inits)),
        "success_threshold": thr, "success_rate": success_rate,
        "n_success": int(success.sum()), "n_better_than_init": n_better,
        "ttest_t": float(t_stat) if t_stat is not None else None,
        "ttest_p_onesided": float(t_p_one) if t_p_one is not None else None,
        "wilcoxon_stat": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p_onesided": float(w_p) if w_p is not None else None,
        "theta_dispersion_normstd": theta_disp,
        "theta_dispersion_perdim": theta_disp_perdim,
        "behav_dispersion_std": behav_disp,
        "manifold_ratio_param_over_behav": manifold_ratio,
        "manifold_pca_eff_dim_participation": eff_dim,
        "manifold_pca_var_ratio": pca_var_ratio,
        "trials": trials, "elapsed_sec": time.time() - t0,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] {args.method} N={args.N} "
          f"d_rec={summary['d_rec_mean']:.4f}+/-{summary['d_rec_std']:.4f} "
          f"CI{summary['d_rec_ci95']} success={success_rate:.0%} "
          f"ttest_p={summary['ttest_p_onesided']} wilcoxon_p={summary['wilcoxon_p_onesided']} "
          f"manifold_ratio={manifold_ratio:.3f} eff_dim={eff_dim} "
          f"({summary['elapsed_sec']/60:.1f} min) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
