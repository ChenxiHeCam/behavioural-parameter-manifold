"""OWMD N2 real-data finetune entry for BAAIWorm full5 recovery.

Reuses the existing recovery infra in
/root/BAAIWorm-main/recovery/scripts/run_full5_multicond.py
(MiniBatchSPSA + build_conditions + cached baselines) but replaces the
per-condition TrajectoryLoss with an OWMDLoss that scores any simulated
trajectory against the *real* OWMD N2 6-metric fingerprint
(forward/backward/turn state props, reversal_rate, turn_magnitude,
body_rhythm_hz).

Loss = mean over the 6 normalised abs-diffs to OWMD per-worm means.
Same loss instance is used for every cond (target is a behavioural
fingerprint, cond-averaged).

Outputs:
  /root/baaiworm_owmd_finetune/result.pkl + result.json
  /root/baaiworm_owmd_finetune.log         (stdout from screen)
  /root/baaiworm_owmd_finetune.json        (final summary)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")
sys.path.insert(0, "/root/BAAIWorm-main/recovery/scripts")

import run_full5_multicond as mc  # noqa: E402
from recovery.utils.io_utils import save_pickle, save_json  # noqa: E402
from recovery.evaluation.behavior_metrics import (  # noqa: E402
    compute_forward_speed,
    compute_body_wave,
    detect_zigzag,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("owmd_ft")

OWMD_KEYS = [
    "forward_state_prop",
    "backward_state_prop",
    "turn_state_prop",
    "reversal_rate",
    "turn_magnitude",
    "body_rhythm_hz",
]


def load_owmd_target(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    per_worm = data["per_worm"]
    target = {}
    for k in OWMD_KEYS:
        vals = [w[k] for w in per_worm if k in w]
        target[k] = float(np.mean(vals)) if vals else 0.0
    target["_n_worms"] = len(per_worm)
    return target


def extract_owmd_from_sim(traj: dict, dt: float = 1.0) -> Dict[str, float]:
    """Compute the OWMD 6 metrics from a BAAIWorm simulated trajectory.

    traj expected to expose rel_x, rel_z (T,17) at minimum.
    States classified per-frame from centroid velocity + heading change:
      forward  : centroid speed > FWD_THR and |dheading| < TURN_THR
      backward : centroid speed > FWD_THR but body-tail vec sign reversed
      turn     : |dheading| >= TURN_THR
    reversal_rate := zigzag.n_reversals / T
    turn_magnitude := mean of |dheading| restricted to turn frames
    body_rhythm_hz := dominant freq from compute_body_wave
    """
    out = {k: 0.0 for k in OWMD_KEYS}
    if "rel_x" not in traj or "rel_z" not in traj:
        return out
    rx = np.asarray(traj["rel_x"])
    rz = np.asarray(traj["rel_z"])
    if rx.ndim != 2 or rx.shape[0] < 5:
        return out

    # Centroid speed
    cx = np.mean(rx, axis=1)
    cz = np.mean(rz, axis=1)
    dx = np.diff(cx)
    dz = np.diff(cz)
    speed = np.sqrt(dx * dx + dz * dz) / dt
    heading = np.arctan2(dz, dx)
    dh = np.diff(heading)
    dh = np.arctan2(np.sin(dh), np.cos(dh))  # wrap

    # Body-axis sign at each frame (head-tail along x in body frame)
    # rel_x[:,0] is head, rel_x[:,-1] is tail in body frame
    body_axis_x = rx[:, 0] - rx[:, -1]
    # Velocity dotted with body axis sign over (T-1) frames
    body_axis_x_avg = 0.5 * (body_axis_x[:-1] + body_axis_x[1:])
    forward_dot = dx * body_axis_x_avg

    FWD_THR = 1e-4
    TURN_THR = 0.08  # rad / step
    T = len(speed)
    if T == 0:
        return out

    moving = speed > FWD_THR
    # dh has length T-1; align by padding
    dh_full = np.zeros(T)
    dh_full[1:] = dh
    is_turn = moving & (np.abs(dh_full) >= TURN_THR)
    is_fwd = moving & ~is_turn & (forward_dot >= 0)
    is_bwd = moving & ~is_turn & (forward_dot < 0)

    out["forward_state_prop"] = float(is_fwd.mean())
    out["backward_state_prop"] = float(is_bwd.mean())
    out["turn_state_prop"] = float(is_turn.mean())

    # Reversals: sign flips in forward_dot
    sign_changes = np.sum(np.diff(np.sign(forward_dot)) != 0)
    out["reversal_rate"] = float(sign_changes) / float(max(T, 1))

    if is_turn.any():
        out["turn_magnitude"] = float(np.mean(np.abs(dh_full[is_turn])))
    else:
        out["turn_magnitude"] = float(np.mean(np.abs(dh_full)))

    # Body rhythm via existing helper
    try:
        bw = compute_body_wave(rx, rz, dt=dt)
        out["body_rhythm_hz"] = float(bw.get("dominant_freq", 0.0))
    except Exception:
        out["body_rhythm_hz"] = 0.0
    return out


class OWMDLoss:
    """Behavioural fingerprint loss vs OWMD N2 per-worm means.

    Same interface as TrajectoryLoss (.compute(traj) -> (scalar, details)),
    so MiniBatchSPSA's `train_loss_fns` / `val_loss_fns` lists accept it.
    """

    # Normalisation scales (rough OWMD magnitudes -> unit-ish)
    SCALES = {
        "forward_state_prop": 0.3,
        "backward_state_prop": 0.15,
        "turn_state_prop": 0.10,
        "reversal_rate": 5e-4,
        "turn_magnitude": 5e-3,
        "body_rhythm_hz": 0.5,
    }

    def __init__(self, target: Dict[str, float]):
        self.target = target
        self.w = {k: 1.0 / len(OWMD_KEYS) for k in OWMD_KEYS}

    def compute(self, recovered_traj):
        feats = extract_owmd_from_sim(recovered_traj)
        details = {}
        total = 0.0
        for k in OWMD_KEYS:
            scale = self.SCALES[k]
            diff = abs(self.target.get(k, 0.0) - feats.get(k, 0.0))
            normd = min(diff / scale, 5.0)
            details[k] = float(normd)
            total += self.w[k] * normd
        details["_feats"] = feats
        return float(total), details


def install_owmd_loss(owmd_target: Dict[str, float], orig_setup):
    """Wrap MiniBatchMixin.setup_minibatch so every loss fn becomes OWMDLoss."""
    def patched(self, train_pairs, val_pairs, batch_size, seed):
        orig_setup(self, train_pairs, val_pairs, batch_size, seed)
        n_tr = len(self.train_loss_fns)
        n_val = len(self.val_loss_fns)
        self.train_loss_fns = [OWMDLoss(owmd_target) for _ in range(n_tr)]
        self.val_loss_fns = [OWMDLoss(owmd_target) for _ in range(n_val)]
        logger.info("OWMDLoss installed on %d train + %d val conds", n_tr, n_val)
    return patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owmd-target", default="/root/owmd_n2_real_features.json")
    ap.add_argument("--init-from",
                    default="/root/BAAIWorm-main/recovery/output/phase2/"
                            "full5_phase3_r1_syn_s202/result.pkl")
    ap.add_argument("--out-name", default="baaiworm_owmd_finetune")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--train-foods", type=int, default=50)
    ap.add_argument("--train-starts", type=int, default=8)
    ap.add_argument("--val-foods", type=int, default=20)
    ap.add_argument("--val-starts", type=int, default=4)
    ap.add_argument("--finetune-foods", type=int, default=25)
    ap.add_argument("--finetune-starts", type=int, default=6)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spsa-a0", type=float, default=0.005)
    ap.add_argument("--spsa-c0", type=float, default=0.02)
    args = ap.parse_args()

    owmd_target = load_owmd_target(args.owmd_target)
    logger.info("OWMD target loaded (n_worms=%d): %s",
                owmd_target.get("_n_worms", 0),
                {k: round(owmd_target[k], 5) for k in OWMD_KEYS})

    # Monkey-patch the loss installation inside MiniBatchMixin
    orig_setup = mc._MiniBatchMixin.setup_minibatch
    mc._MiniBatchMixin.setup_minibatch = install_owmd_loss(owmd_target, orig_setup)

    # Load init weights
    with open(args.init_from, "rb") as f:
        prev = pickle.load(f)
    init_w = prev.get("recovered_weights")
    prev_loss = prev.get("best_loss", "?")
    logger.info("Warm-start %s prev_best=%s val=%s",
                args.init_from, prev_loss,
                prev.get("val_summary", {}).get("mean") if prev.get("val_summary") else None)

    params = mc.FULL5
    n_foods_total = args.train_foods + args.val_foods + args.finetune_foods
    n_starts_total = max(args.train_starts, args.val_starts, args.finetune_starts)
    logger.info("Loading conditions...")
    train_pairs = mc.build_conditions(0, args.train_foods, args.train_starts,
                                       args.n_steps, n_foods_total, n_starts_total)
    val_pairs = mc.build_conditions(args.train_foods,
                                     args.train_foods + args.val_foods,
                                     args.val_starts, args.n_steps,
                                     n_foods_total, n_starts_total)
    logger.info("train=%d val=%d", len(train_pairs), len(val_pairs))
    if len(train_pairs) < args.batch_size:
        logger.error("not enough train conds %d < batch %d",
                     len(train_pairs), args.batch_size)
        return

    opt = mc.MiniBatchSPSA(
        perturbed_weights=init_w,
        train_pairs=train_pairs, val_pairs=val_pairs,
        param_types=params,
        batch_size=args.batch_size, seed=args.seed,
        config={"a0": args.spsa_a0, "c0": args.spsa_c0,
                "max_fevals": args.iters * 4,
                "sim_steps": args.n_steps, "seed": args.seed},
    )

    t0 = time.time()
    result = opt.optimize(n_iterations=args.iters)
    elapsed = time.time() - t0
    val_summary = opt.final_val_loss()
    if val_summary:
        logger.info("Final VAL OWMD: mean=%.4f std=%.4f n=%d",
                    val_summary["mean"], val_summary["std"], val_summary["n"])

    out_dir = os.path.join("/root", args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    result.update(dict(
        scale=1.0, seed=args.seed,
        method="owmd_finetune_spsa",
        param_types=params, full5=True,
        init_from=args.init_from,
        init_from_loss=prev_loss,
        n_train_conds=len(train_pairs),
        n_val_conds=len(val_pairs),
        val_summary=val_summary,
        time_seconds=elapsed,
        owmd_target={k: owmd_target[k] for k in OWMD_KEYS},
    ))
    save_pickle(result, os.path.join(out_dir, "result.pkl"))
    save_json({k: v for k, v in result.items()
               if k not in ("recovered_weights", "loss_history")},
              os.path.join(out_dir, "result.json"))

    summary = {
        "out_dir": out_dir,
        "best_train_loss": float(result.get("best_loss", -1)),
        "val_mean": val_summary["mean"] if val_summary else None,
        "val_n": val_summary["n"] if val_summary else None,
        "owmd_target": {k: owmd_target[k] for k in OWMD_KEYS},
        "iters": args.iters,
        "batch_size": args.batch_size,
        "init_from": args.init_from,
        "elapsed_sec": elapsed,
    }
    with open("/root/baaiworm_owmd_finetune.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("DONE %s best_train=%.4f val=%s elapsed=%.0fs",
                args.out_name, result.get("best_loss", -1),
                f"{val_summary['mean']:.4f}" if val_summary else "?",
                elapsed)


if __name__ == "__main__":
    main()
