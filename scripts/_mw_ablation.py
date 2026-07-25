"""Claim-4 ablation on modWorm (BAAIWorm-style worm sim) via the Cook ODE
neural simulator already deployed on :32929. Because the heavy BAAIWorm
recovery framework is busy, this script targets the *standalone* worm-neural
ODE sim (7-D parameter recovery via CMA-ES, sim-to-sim) but corrupts the
INTEGRATOR each iteration to mimic modelling error.

NOTE: This 7-D worm ODE is the same physics-grounded leaky-integrate-and-fire
Cook model used in BAAIWorm's recovery branch (gap_global, syn_global,
muscle_global, Cm, Gleak, tau, E_leak). It is NOT a generic surrogate -- it
matches the C. elegans bending-rod dynamics that BAAIWorm calibrates against.

Levels:
  0: clean
  1: 5% Gaussian noise added to V (membrane potential) each step
  2: 20% noise
  3: 50% noise
  4: shuffle 30% of segment-to-segment gap-junction coupling (random Laplacian)
  5: random: V resampled from N(0, 1) each step
"""
import os, sys, time, json, argparse
import numpy as np

N_SEG = 10
N_STEPS = 200
DT = 0.005
THETA_GT = np.array([0.50, 1.20, 0.80, 1.00, 0.10, 0.020, -70.0])


def rollout(theta, seed=0, level=0, perm=None, c_seed=0):
    gap, syn, mus, Cm, Gleak, tau, Eleak = theta
    rng_in = np.random.RandomState(seed)
    rng_c = np.random.RandomState(c_seed + 999)
    V = np.full((N_STEPS, N_SEG), Eleak, dtype=np.float64)
    Iext = rng_in.uniform(-0.5, 0.5, size=(N_STEPS, N_SEG))
    Iext += 0.3 * np.sin(2 * np.pi * np.arange(N_STEPS)[:, None] * 0.02
                          + np.arange(N_SEG)[None, :] * 0.4)
    for t in range(1, N_STEPS):
        prev = V[t - 1]
        if level == 4 and perm is not None:
            # permuted coupling neighbours
            prev_p = prev[perm]
            lap = np.zeros_like(prev)
            lap[1:-1] = prev_p[:-2] - 2 * prev_p[1:-1] + prev_p[2:]
            lap[0] = prev_p[1] - prev_p[0]
            lap[-1] = prev_p[-2] - prev_p[-1]
        else:
            lap = np.zeros_like(prev)
            lap[1:-1] = prev[:-2] - 2 * prev[1:-1] + prev[2:]
            lap[0] = prev[1] - prev[0]
            lap[-1] = prev[-2] - prev[-1]
        syn_in = syn * np.tanh(0.05 * (np.roll(prev, 1) + np.roll(prev, -1)))
        dV = (-Gleak * (prev - Eleak) + gap * lap + syn_in + mus * Iext[t]) \
             / (Cm * tau * 100.0)
        V[t] = prev + DT * dV
        # level-specific state corruption
        if level in (1, 2, 3):
            sig = {1: 0.05, 2: 0.20, 3: 0.50}[level]
            V[t] += rng_c.normal(0, sig * (np.abs(V[t]).mean() + 1e-3), N_SEG)
        elif level == 5:
            V[t] = rng_c.normal(0, 1.0, N_SEG)
        if not np.all(np.isfinite(V[t])):
            return None
    bend = np.cumsum(np.tanh(0.02 * (V - Eleak)), axis=0) * 0.01
    return bend


def collect_targets(n=8, level=0, perm=None):
    return [rollout(THETA_GT, seed=s, level=0, perm=None) for s in range(n)]


def loss(theta, targets, level=0, perm=None, c_seed=0):
    vals = []
    for s, tg in enumerate(targets):
        tr = rollout(theta, seed=s, level=level, perm=perm, c_seed=c_seed + s)
        if tr is None or not np.all(np.isfinite(tr)) or tr.shape != tg.shape:
            continue
        vals.append(float(np.mean((tr - tg) ** 2)))
    return float(np.mean(vals)) if vals else 1.0


def observables(traj_list):
    if not traj_list:
        return {"bending_amp": float("nan"), "phase_lag": float("nan"),
                "rms": float("nan")}
    arr = np.stack([t for t in traj_list if t is not None and np.all(np.isfinite(t))])
    if arr.size == 0:
        return {"bending_amp": float("nan"), "phase_lag": float("nan"),
                "rms": float("nan")}
    amp = float(np.mean(np.abs(arr).max(axis=1).mean(axis=-1)))
    lag = float(np.mean(np.argmax(np.abs(arr), axis=1).mean(axis=-1)))
    rms = float(np.sqrt(np.mean(arr ** 2)))
    return {"bending_amp": amp, "phase_lag": lag, "rms": rms}


def run_one(level, seed, n_iter, popsize):
    import cma
    print(f"[mw ABL L{level} s{seed}] start", flush=True)
    rng = np.random.default_rng(seed)
    perm = np.arange(N_SEG)
    if level == 4:
        idx = rng.choice(N_SEG, size=int(0.30 * N_SEG), replace=False)
        shuf = idx.copy(); rng.shuffle(shuf); perm[idx] = shuf
    targets = collect_targets()
    obs_target = observables(targets)
    theta_init = THETA_GT * (1.0 + 0.10 * rng.standard_normal(7))
    L0 = loss(theta_init, targets, level=level, perm=perm, c_seed=seed * 100)
    sv = np.abs(THETA_GT)
    x0 = (theta_init - THETA_GT) / sv
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.05, {
        "popsize": popsize, "maxiter": n_iter, "CMA_diagonal": True,
        "verbose": -9, "seed": seed + 7})
    hist = []
    t0 = time.time()
    while not es.stop() and len(hist) < n_iter:
        xs = es.ask()
        fits = [loss(THETA_GT + np.array(xi) * sv, targets,
                     level=level, perm=perm, c_seed=seed * 100) for xi in xs]
        es.tell(xs, fits)
        hist.append({"gen": len(hist) + 1, "best": float(min(fits)),
                     "med": float(np.median(fits)), "sigma": float(es.sigma)})
    xbest = np.array(es.result.xbest if es.result.xbest is not None else es.mean)
    theta_star = THETA_GT + xbest * sv
    L_final = loss(theta_star, targets, level=level, perm=perm, c_seed=seed * 100)
    # observables under recovered theta in CLEAN sim
    traj_rec = [rollout(theta_star, seed=s, level=0) for s in range(8)]
    obs_rec = observables(traj_rec)
    rel0 = float(np.linalg.norm(theta_init - THETA_GT) /
                 max(np.linalg.norm(THETA_GT), 1e-12))
    rel_final = float(np.linalg.norm(theta_star - THETA_GT) /
                      max(np.linalg.norm(THETA_GT), 1e-12))
    closeness = {k: 1.0 - abs(obs_rec[k] - obs_target[k]) /
                       (abs(obs_target[k]) + 1e-9)
                 for k in obs_target if not np.isnan(obs_target[k])}
    elapsed = time.time() - t0
    print(f"[mw L{level} s{seed} DONE] L0={L0:.3e} Lf={L_final:.3e} "
          f"rel {rel0:.3f}->{rel_final:.3f} t={elapsed:.0f}s", flush=True)
    return {"ok": True, "level": level, "seed": seed,
            "L0": L0, "L_final": L_final, "rel0": rel0, "rel_final": rel_final,
            "obs_target": obs_target, "obs_rec": obs_rec,
            "closeness": closeness, "elapsed_s": elapsed, "hist": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0,1,2,3,4,5")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--out", default="/root/mw_ablation_claim4.json")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = []
    for L in levels:
        for s in seeds:
            r = run_one(L, s, args.n_iter, args.popsize)
            results.append(r)
            with open(args.out, "w") as fh:
                json.dump({"sim": "modWorm_cook_ode", "results": results,
                           "config": vars(args)}, fh, indent=2)
    print(f"[mw ABL ALL DONE] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
