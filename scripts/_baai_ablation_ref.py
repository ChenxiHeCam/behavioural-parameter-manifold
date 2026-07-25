"""Claim-4 modeling-ablation: BAAIWorm (real closed-loop sim on :37565).

Mirrors _fg_ablation.py / _mw_ablation.py design but operates on the actual
BAAIWorm SOTA recovery harness (run_with_custom_weights, full5 weights).

Because BAAIWorm's NeuronXCore sim is a black-box C++ binding -- no access to
per-step membrane V state -- corruption is injected in WEIGHT space (which is
exactly the modelling object the user calibrates). This is semantically
equivalent to the modWorm/FlyGym integrator-noise: "loss-time sim disagrees
with target sim".

Levels:
  0: clean              (loss-time = target weights, baseline recovery)
  1: 5% Gaussian noise on syn_weights+gj_weights
  2: 20% noise
  3: 50% noise
  4: shuffle 30% of syn_weights rows (connectivity decoherence)
  5: random weights (~ unusable sim)

5d theta scale (full5: syn, gj, wout, ion, passive). SPSA 10 iter, 1 cond.

Observable closeness on TrajectoryLoss `details` -> chemo / explor / reorient.
"""
from __future__ import annotations
import os, sys, json, copy, time, argparse, logging
import numpy as np

sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")
sys.path.insert(0, "/root/BAAIWorm-main/recovery/scripts")

import run_multicond_refine_queue as mcr
from save_trajectory_dataset import food_grid, start_grid
from recovery.methods.gradient_descent.trajectory_loss import TrajectoryLoss
from recovery.utils.io_utils import load_pickle

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("BAAI_ABL")

SOTA = "/root/BAAIWorm_clone_ready/BAAIWorm-main/recovery/output/phase2/full5_merge_sota_combo_ionpass_T8/result.pkl"
BASELINE_DIR = "/root/BAAIWorm-main/recovery/output/multicond_baseline"
FULL5 = ["syn", "gj", "wout", "ion_channels", "passive_params"]
FOOD_LO, FOOD_HI = 50, 51
N_STARTS_EVAL = 1
N_STEPS = 100

N_ITER = 10
INIT_PERTURB = 0.2
A_SPSA = 0.05
C_SPSA = 0.05
A_SPALL = 5
ALPHA = 0.602
GAMMA = 0.101


def scale_weights(base_w, theta5):
    w = copy.deepcopy(base_w)
    s_syn, s_gj, s_wout, s_ion, s_pas = theta5
    w["syn_weights"] = w["syn_weights"] * s_syn
    w["gj_weights"] = w["gj_weights"] * s_gj
    w["wout"] = w["wout"] * s_wout
    ic = w["ion_channels"]
    if isinstance(ic, dict):
        w["ion_channels"] = {k: (v * s_ion if isinstance(v, np.ndarray) else v)
                             for k, v in ic.items()}
    pp = w["passive_params"]
    if isinstance(pp, dict):
        w["passive_params"] = {k: (v * s_pas if isinstance(v, np.ndarray) else v)
                               for k, v in pp.items()}
    return w


def corrupt_weights(base_w, level, rng):
    """Apply level-specific corruption to the underlying base weights."""
    w = copy.deepcopy(base_w)
    if level == 0:
        return w
    syn = w["syn_weights"]; gj = w["gj_weights"]
    sa = float(np.abs(syn).mean() + 1e-9)
    ga = float(np.abs(gj).mean() + 1e-9)
    if level in (1, 2, 3):
        sig = {1: 0.05, 2: 0.20, 3: 0.50}[level]
        w["syn_weights"] = syn + rng.normal(0, sig * sa, syn.shape)
        w["gj_weights"] = gj + rng.normal(0, sig * ga, gj.shape)
    elif level == 4:
        # shuffle 30% of rows of syn_weights and gj_weights
        nr = syn.shape[0]
        n_sh = max(1, int(0.30 * nr))
        idx = rng.choice(nr, size=n_sh, replace=False)
        shuf = idx.copy(); rng.shuffle(shuf)
        syn2 = syn.copy(); syn2[idx] = syn[shuf]
        w["syn_weights"] = syn2
        nr2 = gj.shape[0]
        n_sh2 = max(1, int(0.30 * nr2))
        idx2 = rng.choice(nr2, size=n_sh2, replace=False)
        shuf2 = idx2.copy(); rng.shuffle(shuf2)
        gj2 = gj.copy(); gj2[idx2] = gj[shuf2]
        w["gj_weights"] = gj2
    elif level == 5:
        w["syn_weights"] = rng.normal(0, sa, syn.shape)
        w["gj_weights"] = rng.normal(0, ga, gj.shape)
    return w


def build_cond_cache():
    foods = food_grid(95)
    starts = start_grid(8)
    cache = []
    for fi in range(FOOD_LO, FOOD_HI):
        for si in range(N_STARTS_EVAL):
            path = os.path.join(BASELINE_DIR,
                                f"food{fi:02d}_start{si:02d}_steps{N_STEPS}.pkl")
            if not os.path.exists(path):
                continue
            payload = load_pickle(path)
            target = payload["trajectory"]
            s, o = starts[si]
            cond = {"food_idx": fi, "start_idx": si, "food": foods[fi],
                    "start": s, "orientation": o}
            cache.append((fi, si, target, cond))
    return cache


def details_to_obs(d):
    """Map TrajectoryLoss details -> chemo / explor / reorient / curvature."""
    chemo = float(d.get("zigzag", 0.0) + d.get("amplitude", 0.0))
    explor = float(d.get("speed", 0.0) + d.get("speed_std", 0.0))
    reorient = float(d.get("reversals", 0.0))
    curv = float(d.get("head_curvature", d.get("amplitude", 0.0)))
    return {"chemo_zigzag": chemo, "forward_speed": explor,
            "reversal_rate": reorient, "head_curvature": curv}


def eval_one(base_w, theta5, cond_cache):
    w = scale_weights(base_w, theta5)
    totals, obs_acc = [], []
    for (fi, si, target, cond) in cond_cache:
        sim = mcr.make_simulator(cond, N_STEPS)
        try:
            traj = sim.run_with_custom_weights(w, FULL5, n_steps=N_STEPS)
            lfn = TrajectoryLoss(target)
            tot, det = lfn.compute(traj)
            totals.append(float(tot))
            obs_acc.append(details_to_obs(det))
        except Exception as ex:
            log.warning("sim fail: %s", ex)
            totals.append(1.0)
            obs_acc.append({"chemo_zigzag": 1.0, "forward_speed": 1.0,
                            "reversal_rate": 1.0, "head_curvature": 1.0})
    L = float(np.mean(totals))
    keys = obs_acc[0].keys()
    obs_mean = {k: float(np.mean([o[k] for o in obs_acc])) for k in keys}
    return L, obs_mean


def spsa_recover(loss_w, target_w, cond_cache, seed):
    """SPSA on 5d theta. Target traj comes from CLEAN target_w. Loss-time uses
    corrupted loss_w."""
    rng = np.random.RandomState(seed)
    # Pre-compute target observables and target traj cache
    _, obs_target = eval_one(target_w, np.ones(5), cond_cache)
    # Get target trajectories explicitly for MSE (use TrajectoryLoss internal)
    theta = 1.0 + rng.uniform(-INIT_PERTURB, INIT_PERTURB, size=5)
    theta = np.maximum(theta, 0.01)
    best_theta = theta.copy()
    best_L = float("inf")
    hist = []
    for k in range(N_ITER):
        a_k = A_SPSA / (k + 1 + A_SPALL) ** ALPHA
        c_k = C_SPSA / (k + 1) ** GAMMA
        delta = rng.choice([-1, 1], size=5).astype(np.float64)
        tp = np.maximum(theta + c_k * delta, 0.01)
        tm = np.maximum(theta - c_k * delta, 0.01)
        Lp, _ = eval_one(loss_w, tp, cond_cache)
        Lm, _ = eval_one(loss_w, tm, cond_cache)
        g = (Lp - Lm) / (2.0 * c_k * delta)
        gn = np.linalg.norm(g)
        if gn > 10.0: g = g * 10.0 / gn
        theta = np.maximum(theta - a_k * g, 0.01)
        for (th, L) in [(tp, Lp), (tm, Lm)]:
            if L < best_L:
                best_L = L; best_theta = th.copy()
        hist.append({"iter": k, "L_plus": Lp, "L_minus": Lm,
                     "best_L": best_L, "theta": theta.tolist()})
        log.info("[s%d k%d] L+=%.4f L-=%.4f best=%.4f", seed, k, Lp, Lm, best_L)
    return best_theta, best_L, hist, obs_target


def run_one(level, seed, base_w, cond_cache):
    log.info("=== L%d s%d START ===", level, seed)
    rng = np.random.default_rng(seed * 100 + level)
    # Target = clean sim with slightly perturbed theta (matches FG design)
    loss_w = corrupt_weights(base_w, level, rng)
    target_w = base_w  # clean

    L0, obs0 = eval_one(loss_w, np.ones(5), cond_cache)
    t0 = time.time()
    theta_star, L_final, hist, obs_target = spsa_recover(
        loss_w, target_w, cond_cache, seed)
    # Recover-clean: behaviour under best theta in CLEAN sim
    L_clean, obs_rec_clean = eval_one(target_w, theta_star, cond_cache)
    closeness = {}
    for k in obs_target:
        denom = abs(obs_target[k]) + 1e-9
        closeness[k] = float(1.0 - min(1.0, abs(obs_rec_clean[k] - obs_target[k]) / denom))
    elapsed = time.time() - t0
    log.info("[L%d s%d DONE] L0=%.4f Lf=%.4f L_clean=%.4f t=%.0fs",
             level, seed, L0, L_final, L_clean, elapsed)
    return {"ok": True, "level": level, "seed": seed,
            "L0": L0, "L_final": L_final, "L_recovered_in_clean": L_clean,
            "theta_star": theta_star.tolist(),
            "obs_target": obs_target, "obs_rec_clean": obs_rec_clean,
            "closeness": closeness, "elapsed_s": elapsed, "hist": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0,1,2,3,4,5")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="/root/baaiworm_ablation_claim4.json")
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    log.info("loading SOTA base weights")
    res = load_pickle(SOTA)
    base_w = res["recovered_weights"]
    cond_cache = build_cond_cache()
    if not cond_cache:
        log.error("no cond cache"); sys.exit(2)
    log.info("cond_cache n=%d", len(cond_cache))

    results = []
    if os.path.exists(args.out):
        try:
            results = json.load(open(args.out)).get("results", [])
            log.info("resume: %d existing cells", len(results))
        except Exception:
            results = []
    done = {(r["level"], r["seed"]) for r in results if r.get("ok")}

    for L in levels:
        for s in seeds:
            if (L, s) in done:
                log.info("skip done L%d s%d", L, s); continue
            try:
                r = run_one(L, s, base_w, cond_cache)
            except Exception as ex:
                log.exception("cell fail L%d s%d", L, s)
                r = {"ok": False, "level": L, "seed": s, "err": str(ex)}
            results.append(r)
            with open(args.out, "w") as fh:
                json.dump({"sim": "BAAIWorm", "results": results,
                           "config": vars(args)}, fh, indent=2)
    log.info("ALL DONE -> %s", args.out)


if __name__ == "__main__":
    main()
