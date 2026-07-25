"""BAAIWorm three-arm centerline dump across MULTIPLE conditions for genuine
distribution (replaces single-cond + noise-replicate surrogate).

Arms:
  random   : syn/gj = randn*0.1 corruption of SOTA shapes (L5-style random init)
  recovered: SOTA recovered full5 weights (= PGOB = expert-grade calibrated ckpt)

For each arm, roll out N_COND conditions (food x start grid), 100 steps,
save per-cond centerline (T,17,3). Output one npz per arm with X (N_COND,T,17,3).
These feed worm_observables() + raw W1 to OWMD X_raw, identical to the
claim_behavior_pattern_match worm pipeline, so unit-matched to the stored cell.
"""
import os, sys, time, copy, json, traceback
import numpy as np

sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")
sys.path.insert(0, "/root/BAAIWorm-main/recovery/scripts")

import run_multicond_refine_queue as mcr
from save_trajectory_dataset import food_grid, start_grid
from recovery.utils.io_utils import load_pickle

SOTA = "/root/BAAIWorm_clone_ready/BAAIWorm-main/recovery/output/phase2/full5_merge_sota_combo_ionpass_T8/result.pkl"
FULL5 = ["syn", "gj", "wout", "ion_channels", "passive_params"]
N_STEPS = 100
OUTDIR = "/root/autodl-tmp"
# condition grid: spread across food and start to get genuine behavioural variety
FOODS = [20, 50, 80]
STARTS = [0, 1, 2, 3]   # start_grid(8) -> use first 4 orientations
# -> 12 conditions per arm (one rollout ~195s -> ~78 min for 2 arms)


def traj_to_xyz(traj):
    if isinstance(traj, dict):
        x = np.asarray(traj["rel_x"], dtype=np.float32)
        y = np.asarray(traj["rel_y"], dtype=np.float32)
        z = np.asarray(traj["rel_z"], dtype=np.float32)
        return np.stack([x, y, z], axis=-1)
    return np.asarray(traj, dtype=np.float32)


def make_random_weights(w_sota, seed=20260613):
    rng = np.random.RandomState(seed)
    w = copy.deepcopy(w_sota)
    for k in ["syn_weights", "gj_weights"]:
        if k in w:
            arr = w[k]
            w[k] = (rng.randn(*arr.shape).astype(arr.dtype) * 0.1)
    return w


def run_arm(w, label):
    foods = food_grid(95)
    starts = start_grid(8)
    wins = []
    nfail = 0
    for fi in FOODS:
        for si in STARTS:
            s, o = starts[si]
            cond = {"food_idx": fi, "start_idx": si, "food": foods[fi],
                    "start": s, "orientation": o}
            try:
                sim = mcr.make_simulator(cond, N_STEPS)
                traj = sim.run_with_custom_weights(w, FULL5, n_steps=N_STEPS)
                arr = traj_to_xyz(traj)
                if arr.ndim == 3 and arr.shape[1] == 17 and np.all(np.isfinite(arr)):
                    # pad/trim to T=100
                    if arr.shape[0] >= 100:
                        wins.append(arr[:100])
                    else:
                        pad = np.repeat(arr[-1:], 100 - arr.shape[0], axis=0)
                        wins.append(np.concatenate([arr, pad], axis=0))
                else:
                    nfail += 1
            except Exception as e:
                nfail += 1
                if nfail <= 3:
                    print(f"  [{label}] cond({fi},{si}) FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
            print(f"  [{label}] done cond({fi},{si}) wins={len(wins)} fail={nfail}", flush=True)
    if not wins:
        return None
    return np.stack(wins, axis=0)  # (N_COND, 100, 17, 3)


def main():
    t0 = time.time()
    res = load_pickle(SOTA)
    w_sota = res["recovered_weights"]
    print(f"[init] SOTA syn={len(w_sota['syn_weights'])} gj={len(w_sota['gj_weights'])}", flush=True)

    w_random = make_random_weights(w_sota)

    for label, w in [("recovered", w_sota), ("random", w_random)]:
        print(f"[arm] {label} starting elapsed={time.time()-t0:.0f}s", flush=True)
        X = run_arm(w, label)
        if X is None:
            print(f"[arm] {label} produced NO windows", flush=True)
            continue
        out = f"{OUTDIR}/baai_three_arm_{label}.npz"
        np.savez(out, X=X)
        print(f"[arm] {label} SAVED {X.shape} -> {out} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"[DONE] elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
