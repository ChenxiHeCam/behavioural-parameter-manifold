"""Task 1 (autonomous 4-sim direct-real coverage):
modWorm direct-OWMD calibration via SPSA.

Method:
  - Start θ0 = perturb(THETA_GT, scale=1.0, seed=123)  (same as default arm)
  - Loss(θ) = sum_obs W1(sim_obs_dist_MAD, real_obs_dist_MAD)  for 6 observables
    computed over n_rollout=24 short rollouts (cheap), MAD-normalized to OWMD scale.
  - SPSA, n_iter=40, a=0.05, c=0.04, alpha=0.602, gamma=0.101
  - Then evaluate Type 4 = direct ckpt with N=100 rollout -> per-observable W1 vs real.
  - Final JSON has 4-way: default/PGOB/direct/real for 6 observables + trajectory_W1
    summary (speed_proxy + turn_rate_proxy as the headline "trajectory W1" pair).

Output: /root/_mw_direct_owmd_spsa_results.json
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/root')
import modworm_recovery_30d as m
from scipy.stats import wasserstein_distance

T0 = time.time()
RNG = np.random.default_rng(20260611)

OBS_NAMES = ["bend_amp", "bend_mean", "head_freq", "propagation",
             "speed_proxy", "turn_rate_proxy"]

with open("/root/_owmd_observables.json") as f:
    OWMD = json.load(f)
REAL = {k: np.array(v, float) for k, v in OWMD["real_dist"].items()}
REAL_NORM = {}
REAL_NORM_STATS = {}
for k in OBS_NAMES:
    a = REAL[k]
    med = float(np.median(a)); mad = float(np.median(np.abs(a - med))) + 1e-9
    REAL_NORM_STATS[k] = (med, mad)
    REAL_NORM[k] = (a - med) / mad


def observables(bend):
    T, K = bend.shape
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
    return {
        "bend_amp":        float(amp.mean()),
        "bend_mean":       float(bend.mean()),
        "head_freq":       head_freq,
        "propagation":     prop,
        "speed_proxy":     float(np.abs(dbend).mean()),
        "turn_rate_proxy": float(np.abs(np.diff(mid)).mean()),
    }


def collect_obs(theta, n_rollout, seed_base):
    rows = []; fails = 0
    for i in range(n_rollout):
        out = m.rollout(theta, seed=seed_base + i)
        if out is None:
            fails += 1
            continue
        rows.append(observables(out))
    return rows, fails


def w1_loss(theta, n_rollout=24, seed_base=70000):
    rows, fails = collect_obs(theta, n_rollout, seed_base)
    if len(rows) < 4:
        return 1e3
    tot = 0.0
    for k in OBS_NAMES:
        a = np.array([r[k] for r in rows], float)
        med, mad = REAL_NORM_STATS[k]
        an = (a - med) / mad
        an = an[np.isfinite(an)]
        if len(an) < 2:
            tot += 5.0
            continue
        tot += float(wasserstein_distance(an, REAL_NORM[k][np.isfinite(REAL_NORM[k])]))
    return tot


# Init: same as "default" arm (scale=1.0 perturb)
theta0 = m.perturb(m.THETA_GT, 1.0, seed=123).astype(float)
DIM = theta0.size
scale_vec = np.where(m.THETA_GT > 0, np.abs(m.THETA_GT),
                     np.maximum(np.abs(m.THETA_GT), 1e-3))

L0 = w1_loss(theta0, n_rollout=24, seed_base=70000)
print(f"[INIT] dim={DIM} L0={L0:.4f}", flush=True)

# SPSA
N_ITER = 40
a0, c0, A = 0.05, 0.04, 5.0
alpha, gamma = 0.602, 0.101
theta = theta0.copy()
best = (L0, theta.copy())
hist = [{"iter": 0, "loss": L0}]
for it in range(1, N_ITER + 1):
    ak = a0 / (it + A) ** alpha
    ck = c0 / it ** gamma
    delta = (RNG.integers(0, 2, size=DIM) * 2 - 1).astype(float)
    pert = ck * delta * scale_vec
    Lp = w1_loss(theta + pert, n_rollout=20, seed_base=80000 + it * 1000)
    Lm = w1_loss(theta - pert, n_rollout=20, seed_base=80000 + it * 1000 + 500)
    g = (Lp - Lm) / (2.0 * ck * delta) / scale_vec
    theta = theta - ak * scale_vec * g
    if (it % 5) == 0 or it <= 3:
        Lc = w1_loss(theta, n_rollout=24, seed_base=70000)
        hist.append({"iter": it, "loss": Lc, "Lp": Lp, "Lm": Lm})
        if Lc < best[0]:
            best = (Lc, theta.copy())
        print(f"  [spsa it={it:02d}] L={Lc:.4f} Lp={Lp:.4f} Lm={Lm:.4f} ak={ak:.4f} ck={ck:.4f}", flush=True)

Lf = w1_loss(best[1], n_rollout=48, seed_base=90000)
print(f"[FINAL] L0={L0:.4f} best_L={best[0]:.4f} Lf_n48={Lf:.4f}", flush=True)

theta_direct = best[1]

# Now N=100 4-way evaluation (default / PGOB / direct / real)
N_EVAL = 100
print(f"[EVAL] N={N_EVAL}-per-arm 4-way", flush=True)


def eval_arm(name, theta_fn):
    rows = []; fails = 0
    for i in range(N_EVAL):
        th = theta_fn(i)
        out = m.rollout(th, seed=30000 + i)
        if out is None:
            fails += 1; continue
        rows.append(observables(out))
    return rows, fails


def_rows, def_fails = eval_arm("default",
                                lambda i: m.perturb(m.THETA_GT, 1.0, seed=10000 + i))
pgob_rows, pgob_fails = eval_arm("pgob",
                                  lambda i: m.perturb(m.THETA_GT, 0.1, seed=20000 + i))
direct_rows, direct_fails = eval_arm("direct",
                                      lambda i: theta_direct)


def w1_per_obs(rows_a, key):
    a = np.array([r[key] for r in rows_a], float)
    med, mad = REAL_NORM_STATS[key]
    an = (a - med) / mad
    an = an[np.isfinite(an)]
    rn = REAL_NORM[key]; rn = rn[np.isfinite(rn)]
    if len(an) < 2 or len(rn) < 2: return float("nan")
    return float(wasserstein_distance(an, rn))


per_obs = []
for k in OBS_NAMES:
    d_def = w1_per_obs(def_rows, k)
    d_pgob = w1_per_obs(pgob_rows, k)
    d_dir = w1_per_obs(direct_rows, k)
    per_obs.append({
        "observable": k,
        "d_default_vs_real": d_def,
        "d_PGOB_vs_real": d_pgob,
        "d_direct_vs_real": d_dir,
        "real_mean": float(np.mean(REAL[k])),
        "default_mean": float(np.mean([r[k] for r in def_rows])) if def_rows else float("nan"),
        "pgob_mean":    float(np.mean([r[k] for r in pgob_rows])) if pgob_rows else float("nan"),
        "direct_mean":  float(np.mean([r[k] for r in direct_rows])) if direct_rows else float("nan"),
    })

# "Trajectory W1" headline = speed_proxy + turn_rate_proxy (mirror BAAIWorm schema)
def find(name):
    return next(p for p in per_obs if p["observable"] == name)

speed = find("speed_proxy"); turn = find("turn_rate_proxy")
traj_w1 = {
    "speed_default_vs_real": speed["d_default_vs_real"],
    "speed_pgob_vs_real":    speed["d_PGOB_vs_real"],
    "speed_direct_vs_real":  speed["d_direct_vs_real"],
    "turn_default_vs_real":  turn["d_default_vs_real"],
    "turn_pgob_vs_real":     turn["d_PGOB_vs_real"],
    "turn_direct_vs_real":   turn["d_direct_vs_real"],
}

out = {
    "type4_method": f"SPSA {N_ITER} iter on modWorm 30d full θ (dim={DIM}), "
                    f"init from default-scale-1.0 perturb of THETA_GT, "
                    f"loss = sum W1(sim_obs_dist_MAD, OWMD_real_dist_MAD) over 6 observables, "
                    f"n_rollout_per_eval=24",
    "sim": "modWorm_30d",
    "real": "OWMD real (5010 windows, MAD-normalized 6 observables)",
    "dim_theta": DIM,
    "n_iter_spsa": N_ITER,
    "init_loss": L0,
    "best_loss": best[0],
    "final_loss_n48": Lf,
    "n_eval_per_arm": N_EVAL,
    "fails": {"default": def_fails, "pgob": pgob_fails, "direct": direct_fails},
    "trajectory_W1_to_real": traj_w1,
    "per_observable_4way": per_obs,
    "spsa_history": hist,
    "elapsed_sec": time.time() - T0,
}

# Interpretation
sp_pgob = traj_w1["speed_pgob_vs_real"]; sp_dir = traj_w1["speed_direct_vs_real"]
tu_pgob = traj_w1["turn_pgob_vs_real"];  tu_dir = traj_w1["turn_direct_vs_real"]
if not (np.isnan(sp_pgob) or np.isnan(sp_dir)):
    if sp_pgob < sp_dir:
        out["interpretation_speed"] = f"PGOB (speed W1={sp_pgob:.4f}) BEATS direct ({sp_dir:.4f})"
    else:
        out["interpretation_speed"] = f"Direct (speed W1={sp_dir:.4f}) beats PGOB ({sp_pgob:.4f})"
if not (np.isnan(tu_pgob) or np.isnan(tu_dir)):
    if tu_pgob < tu_dir:
        out["interpretation_turn"] = f"PGOB (turn W1={tu_pgob:.4f}) BEATS direct ({tu_dir:.4f})"
    else:
        out["interpretation_turn"] = f"Direct (turn W1={tu_dir:.4f}) beats PGOB ({tu_pgob:.4f})"

# Save direct theta too
np.save("/root/_mw_direct_owmd_theta.npy", theta_direct)
with open("/root/_mw_direct_owmd_spsa_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"DONE elapsed={time.time()-T0:.0f}s")
print(f"  init L={L0:.4f} -> best L={best[0]:.4f}")
print(f"  speed W1: def={traj_w1['speed_default_vs_real']:.4f}  pgob={sp_pgob:.4f}  direct={sp_dir:.4f}")
print(f"  turn  W1: def={traj_w1['turn_default_vs_real']:.4f}  pgob={tu_pgob:.4f}  direct={tu_dir:.4f}")
