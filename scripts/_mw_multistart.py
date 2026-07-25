"""Fixed-theta* multistart recovery on modWorm (30-dim).

Strongest PGOB-sim statistical experiment: fix ONE ground-truth theta*, launch
N independent random initial points (fixed seed list -> reproducible), recover
each via PGOB optimiser (SPSA or CMA-ES) against the SAME theta*-target
trajectory, then report the recovery distribution + statistical significance.

This is impossible in real wet-lab data (no ground truth, individual variation);
only sim has theta* so we can measure parameter error too.

Metrics:
  (a) behaviour distance (W1 over trajectory observables) to theta*-target:
      mean +/- std + bootstrap 95% CI over N independent trials
  (b) recovery success rate: fraction of N trials whose behaviour distance
      drops below a threshold (close to theta* behaviour)
  (c) theta parameter error distribution (ground truth available)
  (d) manifold check: are the N recovered theta scattered in parameter space
      while their behaviour all collapses onto theta*? (param dispersion vs
      behaviour dispersion) -> evidence for non-unique parameter manifold.

Train/test split: optimiser fits TRAIN rollout-seed set; behaviour distance to
theta* is evaluated on a held-out TEST rollout-seed set.

Pure-trajectory observables only (no muscle/neural internals).
Deterministic: rollout seeded; init-point seed list fixed.
"""
import os, sys, time, json, argparse
import numpy as np
from scipy.stats import wasserstein_distance, ttest_1samp, wilcoxon

# ---------------- modWorm 30-dim forward model (embedded, self-contained) ----
N_SEG = 10
N_STEPS = 200
DT = 0.005

THETA_GT = np.concatenate([
    np.full(10, 0.50),                            # gap per segment
    np.full(10, 1.20),                            # syn per segment
    np.array([1.00, 0.10, 0.020, -70.0, 0.05]),  # Cm, Gleak, tau, Eleak, g_in
    np.array([0.80, 0.78, 0.75, 0.72, 0.70]),    # 5 muscle bands
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


# ---------------- trajectory observables (pure-trajectory) -------------------
OBS_NAMES = ["bend_amp", "bend_mean", "head_freq", "propagation",
             "speed_proxy", "turn_rate_proxy"]


def observables(bend):
    mid = bend.mean(axis=1)
    amp = bend.std(axis=0)
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


# behaviour distance between two observable-distributions (sum of per-obs W1,
# normalised by target MAD per observable -> dimensionless)
def behav_dist(obs_a, obs_target, scale_mad):
    if obs_a is None or len(obs_a) == 0:
        return np.nan
    d = 0.0
    for j in range(len(OBS_NAMES)):
        a = obs_a[:, j] / scale_mad[j]
        t = obs_target[:, j] / scale_mad[j]
        d += wasserstein_distance(a, t)
    return float(d / len(OBS_NAMES))


# scalar MSE loss for the optimiser (fit on TRAIN seeds) -- pure trajectory
def loss_mse(theta, target_trajs, seeds):
    vals = []
    for s, tg in zip(seeds, target_trajs):
        tr = rollout(theta, seed=s)
        if tr is None or not np.all(np.isfinite(tr)):
            continue
        vals.append(float(np.mean((tr - tg) ** 2)))
    return float(np.mean(vals)) if vals else 1.0


# ---------------- optimisers -------------------------------------------------
def spsa(theta0, target_trajs, train_seeds, n_iter, seed):
    rng = np.random.default_rng(seed)
    K = len(theta0)
    scale_vec = np.abs(THETA_GT)
    a, c = 0.20, 0.08
    # operate in normalised coordinates x = (theta - GT)/scale_vec
    x = (theta0 - THETA_GT) / scale_vec
    x_best = x.copy()
    L_best = loss_mse(THETA_GT + x * scale_vec, target_trajs, train_seeds)
    for it in range(n_iter):
        ck = c / (it + 1) ** 0.101
        ak = a / (it + 1) ** 0.602
        bk = rng.choice([-1.0, 1.0], size=K)
        xp = x + ck * bk; xm = x - ck * bk
        Lp = loss_mse(THETA_GT + xp * scale_vec, target_trajs, train_seeds)
        Lm = loss_mse(THETA_GT + xm * scale_vec, target_trajs, train_seeds)
        if not (np.isfinite(Lp) and np.isfinite(Lm)):
            continue
        # relative finite-difference gradient (loss spans many orders of
        # magnitude -> normalise by current best to keep steps in trust region)
        ghat = (Lp - Lm) / (2 * ck * max(L_best, 1e-9)) * bk
        # clip per-element step to a trust radius
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
        es.tell(xs, losses)
        gen += 1
    xbest = np.array(es.result.xbest) if es.result.xbest is not None else np.array(es.mean)
    return THETA_GT + xbest * scale_vec


# ---------------- bootstrap CI ----------------------------------------------
def bootstrap_ci(vals, n_boot=10000, alpha=0.05, seed=7):
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if len(vals) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(vals, size=len(vals), replace=True).mean())
             for _ in range(n_boot)]
    return [float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--n_iter", type=int, default=40)
    ap.add_argument("--method", choices=["spsa", "cma"], default="cma")
    ap.add_argument("--init_scale", type=float, default=0.30,
                    help="random init perturbation scale (random restart radius)")
    ap.add_argument("--out", default="/root/_mw_multistart_cma.json")
    args = ap.parse_args()

    # train/test rollout-seed split (fixed) -- optimiser fits TRAIN, eval on TEST
    TRAIN_SEEDS = list(range(0, 8))
    TEST_SEEDS = list(range(100, 116))   # held out
    # target trajectories (theta*) for the optimiser's train loss
    target_trajs_train = [rollout(THETA_GT, seed=s) for s in TRAIN_SEEDS]
    # target observable distribution (theta*) on TEST seeds
    obs_target_test = collect_obs(THETA_GT, TEST_SEEDS)
    scale_mad = np.array([
        (np.median(np.abs(obs_target_test[:, j] - np.median(obs_target_test[:, j]))) + 1e-9)
        for j in range(len(OBS_NAMES))])

    # success threshold: 20% of the *default-init* baseline behaviour distance
    # (so "success" = recovered behaviour is within the bottom 20% of the
    #  init-to-target gap). Computed from baseline trials below.
    INIT_SEEDS = list(range(2000, 2000 + args.N))   # fixed reproducible init seeds

    print(f"[mw-multistart] method={args.method} N={args.N} n_iter={args.n_iter} "
          f"init_scale={args.init_scale}", flush=True)

    trials = []
    t0 = time.time()
    for i, iseed in enumerate(INIT_SEEDS):
        rng = np.random.RandomState(iseed)
        theta_init = THETA_GT * (1.0 + args.init_scale * rng.randn(len(THETA_GT)))
        obs_init_test = collect_obs(theta_init, TEST_SEEDS)
        d_init = behav_dist(obs_init_test, obs_target_test, scale_mad)
        relerr_init = float(np.mean(np.abs((theta_init - THETA_GT) / THETA_GT)))

        ti = time.time()
        if args.method == "spsa":
            theta_rec = spsa(theta_init, target_trajs_train, TRAIN_SEEDS, args.n_iter, iseed)
        else:
            theta_rec = cma(theta_init, target_trajs_train, TRAIN_SEEDS, args.n_iter, iseed)

        obs_rec_test = collect_obs(theta_rec, TEST_SEEDS)
        d_rec = behav_dist(obs_rec_test, obs_target_test, scale_mad)
        relerr_rec = float(np.mean(np.abs((theta_rec - THETA_GT) / THETA_GT)))
        improvement = (d_init - d_rec) / d_init * 100.0 if d_init > 0 else np.nan

        trials.append({
            "init_seed": int(iseed),
            "d_init": d_init, "d_rec": d_rec, "improvement_pct": improvement,
            "relerr_init": relerr_init, "relerr_rec": relerr_rec,
            "theta_rec": theta_rec.tolist(),
            "wall_sec": time.time() - ti,
        })
        print(f"[trial {i+1:02d}/{args.N}] d_init={d_init:.4f} d_rec={d_rec:.4f} "
              f"impr={improvement:5.1f}% relerr {relerr_init:.3f}->{relerr_rec:.3f} "
              f"({time.time()-ti:.1f}s)", flush=True)

    # ---------- statistics ----------
    d_recs = np.array([t["d_rec"] for t in trials], float)
    d_inits = np.array([t["d_init"] for t in trials], float)
    relerr_recs = np.array([t["relerr_rec"] for t in trials], float)
    relerr_inits = np.array([t["relerr_init"] for t in trials], float)

    # success threshold = 20% of the median init behaviour distance
    thr = 0.20 * float(np.median(d_inits))
    success = d_recs < thr
    success_rate = float(success.mean())

    # paired test: recovered behaviour distance significantly < init baseline?
    diffs = d_inits - d_recs   # positive = recovery improved
    try:
        t_stat, t_p = ttest_1samp(diffs, 0.0)
        t_p_one = t_p / 2 if t_stat > 0 else 1 - t_p / 2  # one-sided (recovery better)
    except Exception:
        t_stat, t_p_one = None, None
    try:
        w_stat, w_p = wilcoxon(d_inits, d_recs, alternative="greater")
    except Exception:
        w_stat, w_p = None, None
    # sign test
    n_better = int((d_recs < d_inits).sum())

    # manifold dispersion: spread of recovered theta in param space vs spread of
    # their behaviour. high param dispersion + low behaviour dispersion -> manifold
    thetas = np.array([t["theta_rec"] for t in trials])  # (N,30)
    theta_disp = float(np.mean(np.std(thetas / np.abs(THETA_GT), axis=0)))  # mean normalised std per dim
    behav_disp = float(np.std(d_recs))
    # also: pairwise behaviour distance among recovered solutions vs to-target dist
    # ratio param_dispersion / behaviour_distance_to_target -> manifold breadth
    mean_d_rec = float(np.nanmean(d_recs))
    manifold_ratio = theta_disp / (mean_d_rec + 1e-9)

    summary = {
        "sim": "modWorm_30d",
        "method": args.method.upper(),
        "N": args.N,
        "n_iter": args.n_iter,
        "init_scale": args.init_scale,
        "train_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS,
        "obs_names": OBS_NAMES,
        "theta_gt": THETA_GT.tolist(),
        # behaviour distance to theta*-target
        "d_rec_mean": float(np.nanmean(d_recs)),
        "d_rec_std": float(np.nanstd(d_recs)),
        "d_rec_ci95": bootstrap_ci(d_recs),
        "d_init_mean": float(np.nanmean(d_inits)),
        "d_init_std": float(np.nanstd(d_inits)),
        "d_init_ci95": bootstrap_ci(d_inits),
        # parameter error to theta*
        "relerr_rec_mean": float(np.nanmean(relerr_recs)),
        "relerr_rec_std": float(np.nanstd(relerr_recs)),
        "relerr_rec_ci95": bootstrap_ci(relerr_recs),
        "relerr_init_mean": float(np.nanmean(relerr_inits)),
        # success
        "success_threshold": thr,
        "success_rate": success_rate,
        "n_success": int(success.sum()),
        # significance: recovery vs init baseline
        "n_better_than_init": n_better,
        "ttest_t": float(t_stat) if t_stat is not None else None,
        "ttest_p_onesided": float(t_p_one) if t_p_one is not None else None,
        "wilcoxon_stat": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p_onesided": float(w_p) if w_p is not None else None,
        # manifold
        "theta_dispersion_normstd": theta_disp,
        "behav_dispersion_std": behav_disp,
        "manifold_ratio_param_over_behav": manifold_ratio,
        "trials": trials,
        "elapsed_sec": time.time() - t0,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] {args.method} N={args.N} "
          f"d_rec={summary['d_rec_mean']:.4f}+/-{summary['d_rec_std']:.4f} "
          f"CI{summary['d_rec_ci95']} success={success_rate:.0%} "
          f"ttest_p={summary['ttest_p_onesided']} "
          f"manifold_ratio={manifold_ratio:.2f} "
          f"({summary['elapsed_sec']/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
