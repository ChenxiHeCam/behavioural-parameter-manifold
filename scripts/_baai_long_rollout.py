"""Claim-9 time-extrapolation: BAAIWorm long rollout (>300 frames).

Memory note: claim 9 BAAIWorm previously only tested 100->300 (NEURON budget).
This fills the untested >300 segment: roll out the recovered SOTA weights for
T in {100, 300, 1000, 2000} at fixed cond, compute pure-trajectory observables
on sliding/cumulative windows, and check observable stability (drift) as T grows.

If PGOB-recovered behavior is time-extrapolable, the per-window observables
computed over [0:100], [0:300], [0:1000], [0:2000] stay in the same range
(low drift). Pure-trajectory observables only [[pure-trajectory-only]];
behavior closeness / stability is the metric [[behavior-closeness-metric]].
Sim is bit-perfect deterministic [[sim-deterministic]] so single rollout/T.
"""
import os, sys, json, time, copy
import numpy as np

sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")
sys.path.insert(0, "/root/BAAIWorm-main/recovery/scripts")

import run_multicond_refine_queue as mcr
from save_trajectory_dataset import food_grid, start_grid
from recovery.utils.io_utils import load_pickle

SOTA = "/root/BAAIWorm_clone_ready/BAAIWorm-main/recovery/output/phase2/full5_merge_sota_combo_ionpass_T8/result.pkl"
FULL5 = ["syn", "gj", "wout", "ion_channels", "passive_params"]
TS = [100, 300, 1000, 2000]
FOOD, START = 50, 0
OUT = "/root/autodl-tmp/baai_long_rollout_claim9.json"


N_INIT = 30  # init frames prepended to world_head_location


def traj_to_xy(traj):
    # BAAIWorm run_closed_loop returns a dict; world_head_location (Ninit+T, 3)
    # is the worm's head path in world coords -- a pure positional trajectory
    # (no muscle/neural signals). Drop the init frames.
    if isinstance(traj, dict):
        wh = traj.get("world_head_location")
        if wh is None:
            return np.zeros((0, 2))
        wh = np.asarray(wh, dtype=np.float64)
        if wh.ndim != 2 or wh.shape[1] < 2:
            return np.zeros((0, 2))
        xy = wh[:, :2]
        if len(xy) > N_INIT:
            xy = xy[N_INIT:]
        return xy
    a = np.asarray(traj, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] >= 2:
        return a[:, :2]
    return np.column_stack([a[:, 0], np.zeros(len(a))])


def window_obs(xy):
    if len(xy) < 3 or not np.all(np.isfinite(xy)):
        return None
    d = np.diff(xy, axis=0)
    seg = np.linalg.norm(d, axis=1)
    head = np.arctan2(d[:, 1], d[:, 0])
    dh = np.diff(head)
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    return {"mean_speed": float(seg.mean()),
            "speed_std": float(seg.std()),
            "turn": float(np.abs(dh).mean()) if len(dh) else 0.0,
            "path_per_step": float(seg.sum() / max(1, len(seg))),
            "net_disp_per_step": float(np.linalg.norm(xy[-1] - xy[0]) / len(xy))}


def main():
    res = load_pickle(SOTA)
    w = res["recovered_weights"]
    foods = food_grid(95); starts = start_grid(8)
    s, o = starts[START]
    cond = {"food_idx": FOOD, "start_idx": START, "food": foods[FOOD],
            "start": s, "orientation": o}

    out = {"sim": "BAAIWorm_closed_loop", "cond": [FOOD, START], "Ts": TS,
           "rollouts": {}}
    # one long T=2000 rollout, then slice prefixes (deterministic) to compare.
    # But closed-loop n_steps differs per call; do separate calls per T so each
    # is an independent honest rollout (init identical, deterministic).
    base_obs = None
    for T in TS:
        t0 = time.time()
        try:
            traj = w_run(cond, w, T)
            xy = traj_to_xy(traj)
            full = window_obs(xy)
            # cumulative-prefix observables at fixed checkpoints
            prefixes = {}
            for cp in [100, 300, 1000, 2000]:
                if cp <= len(xy):
                    prefixes[str(cp)] = window_obs(xy[:cp])
            ob = {"T": T, "n_frames": int(len(xy)), "full_obs": full,
                  "prefix_obs": prefixes, "wall_sec": time.time() - t0}
        except Exception as ex:
            ob = {"T": T, "err": str(ex), "wall_sec": time.time() - t0}
        out["rollouts"][str(T)] = ob
        print(f"[T={T}] {ob.get('n_frames','ERR')} frames "
              f"obs={ob.get('full_obs')} wall={ob['wall_sec']:.0f}s", flush=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2)

    # drift: compare full_obs across T to the T=100 reference
    ref = out["rollouts"].get("100", {}).get("full_obs")
    drift = {}
    if ref:
        for T in TS:
            fo = out["rollouts"].get(str(T), {}).get("full_obs")
            if fo:
                drift[str(T)] = {k: float(abs(fo[k] - ref[k]) / (abs(ref[k]) + 1e-9))
                                 for k in ref}
    out["drift_vs_T100"] = drift
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("DONE ->", OUT, flush=True)


def w_run(cond, w, T):
    sim = mcr.make_simulator(cond, T)
    return sim.run_with_custom_weights(w, FULL5, n_steps=T)


if __name__ == "__main__":
    main()
