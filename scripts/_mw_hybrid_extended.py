"""GENUINE modWorm Cook-ODE extended-budget PGOB-hybrid: sim warm-start -> OWMD-real
fine-tune, full d_real(W1-to-OWMD) trajectory.

Corroborates the local surrogate hybrid_extended_budget.py with the real Cook ODE
forward model and real OWMD observable distributions.

Stage 1 (sim warm-start): theta_warm = PGOB-sim point (perturb(THETA_GT, 0.1)) --
  the sim-recovered theta arm. (Same 'pgob' arm as _mw_direct_owmd_spsa.py.)
Stage 2 (OWMD real fine-tune): SPSA toward real OWMD, RELAXED budget:
  - 40 iters, d_real(=sum W1 over 6 OWMD observables) logged EVERY iter on a
    held-out real eval (different seed_base than the train loss)
  - drift reg lam*||theta-theta_warm||^2 + box (theta_warm +- box*scale) +
    grad clip  = divergence guards (prevents the 38x blow-up)
  - NO early stop in stage 2; the full curve is the evidence.
fit on train seeds, judge d_real on held-out eval seeds (train/test split by seed).

Output: /root/_mw_hybrid_extended_results.json
"""
import os, sys, time, json
import numpy as np
sys.path.insert(0, '/root')
import modworm_recovery_30d as m
from scipy.stats import wasserstein_distance

T0 = time.time()
RNG = np.random.default_rng(20260613)
OBS_NAMES = ["bend_amp", "bend_mean", "head_freq", "propagation",
             "speed_proxy", "turn_rate_proxy"]

with open("/root/_owmd_observables.json") as f:
    OWMD = json.load(f)
REAL = {k: np.array(v, float) for k, v in OWMD["real_dist"].items()}
REAL_NORM = {}; REAL_NORM_STATS = {}
# Drop observables whose REAL distribution is degenerate (near-zero MAD): the
# MAD-normalized W1 there is ill-defined (divide-by-~0) and swamps the metric.
# In OWMD, head_freq is a constant (MAD~1e-9) -> excluded honestly.
USE_OBS = []
for k in OBS_NAMES:
    a = REAL[k]
    med = float(np.median(a)); mad = float(np.median(np.abs(a - med)))
    if mad < 1e-6:
        print(f"[obs] dropping '{k}' (degenerate real MAD={mad:.2e})", flush=True)
        continue
    REAL_NORM_STATS[k] = (med, mad + 1e-9)
    REAL_NORM[k] = (a - med) / (mad + 1e-9)
    USE_OBS.append(k)
print(f"[obs] using {len(USE_OBS)} observables: {USE_OBS}", flush=True)


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
    return {"bend_amp": float(amp.mean()), "bend_mean": float(bend.mean()),
            "head_freq": head_freq, "propagation": prop,
            "speed_proxy": float(np.abs(dbend).mean()),
            "turn_rate_proxy": float(np.abs(np.diff(mid)).mean())}


def collect_obs(theta, n_rollout, seed_base):
    rows = []
    for i in range(n_rollout):
        out = m.rollout(theta, seed=seed_base + i)
        if out is None:
            continue
        rows.append(observables(out))
    return rows


def w1_dist(theta, n_rollout, seed_base):
    """d_real: sum of W1 over 6 OWMD observables (MAD-normalized)."""
    rows = collect_obs(theta, n_rollout, seed_base)
    if len(rows) < 4:
        return 1e3
    tot = 0.0
    for k in USE_OBS:
        a = np.array([r[k] for r in rows], float)
        med, mad = REAL_NORM_STATS[k]
        an = (a - med) / mad
        an = an[np.isfinite(an)]
        rn = REAL_NORM[k][np.isfinite(REAL_NORM[k])]
        if len(an) < 2:
            tot += 5.0; continue
        tot += float(wasserstein_distance(an, rn))
    return tot


# ---- Stage 1: sim warm-start = PGOB-sim theta (perturb 0.1 of THETA_GT) -------
theta_warm = m.perturb(m.THETA_GT, 0.1, seed=20000).astype(float)
DIM = theta_warm.size
scale_vec = np.where(m.THETA_GT > 0, np.abs(m.THETA_GT),
                     np.maximum(np.abs(m.THETA_GT), 1e-3))

# train loss = W1 on TRAIN seeds; d_real eval = W1 on held-out EVAL seeds
TRAIN_SEED = 70000
EVAL_SEED = 50000   # disjoint from train
N_TRAIN = 24
N_EVAL = 40

d_warm = w1_dist(theta_warm, N_EVAL, EVAL_SEED)
print(f"[WARM] dim={DIM} d_real(warm,eval)={d_warm:.4f}", flush=True)

# ---- Stage 2: extended OWMD fine-tune, full d_real curve, divergence guards ---
N_ITER = 40
a0, c0, A = 0.05, 0.04, 5.0
alpha, gamma = 0.602, 0.101
LAM = 0.02
BOX = 0.6  # theta_warm +- BOX*scale_vec
lo = theta_warm - BOX * scale_vec
hi = theta_warm + BOX * scale_vec


def L_train(theta, it):
    theta = np.clip(theta, lo, hi)
    base = w1_dist(theta, N_TRAIN, TRAIN_SEED + it * 1000)
    drift = LAM * float(np.sum(((theta - theta_warm) / scale_vec) ** 2))
    return base + drift


theta = theta_warm.copy()
curve = [{"iter": -1, "stage": "warm_start", "d_real_eval": d_warm}]
best_d = d_warm; best_theta = theta.copy(); best_iter = -1
for it in range(1, N_ITER + 1):
    ak = a0 / (it + A) ** alpha
    ck = c0 / it ** gamma
    delta = (RNG.integers(0, 2, size=DIM) * 2 - 1).astype(float)
    pert = ck * delta * scale_vec
    Lp = L_train(theta + pert, it)
    Lm = L_train(theta - pert, it)
    g = (Lp - Lm) / (2.0 * ck * delta) / scale_vec
    gn = np.linalg.norm(g * scale_vec)
    if gn > 50.0:
        g = g * 50.0 / gn
    theta = np.clip(theta - ak * scale_vec * g, lo, hi)
    d_eval = w1_dist(theta, N_EVAL, EVAL_SEED)  # held-out d_real every iter
    curve.append({"iter": it, "stage": "finetune", "d_real_eval": d_eval,
                  "L_train_p": Lp, "L_train_m": Lm})
    if d_eval < best_d:
        best_d, best_theta, best_iter = d_eval, theta.copy(), it
    print(f"  [ft it={it:02d}] d_real_eval={d_eval:.4f} (best {best_d:.4f}) "
          f"ak={ak:.4f}", flush=True)

d_final = w1_dist(theta, N_EVAL, EVAL_SEED)
d_final_hi = w1_dist(best_theta, 80, EVAL_SEED)  # high-N confirm at best

evals = [c["d_real_eval"] for c in curve if "d_real_eval" in c]
min_d = float(np.min(evals)); min_iter = int(np.argmin(evals)) - 1

# verdict (no preset)
eps = 1e-3
dipped = min_d < d_warm - eps
ended_below = d_final < d_warm - eps
if dipped and ended_below:
    verdict = "HYBRID_USEFUL"
elif dipped and not ended_below:
    verdict = "HYBRID_TROUGH_THEN_DRIFT"
else:
    verdict = "HYBRID_DEATH_END"

out = {
    "sim": "modWorm_30d_CookODE",
    "mode": "PGOB-hybrid_extended_budget",
    "real": "OWMD real (MAD-normalized observables; degenerate head_freq dropped)",
    "metric": "d_real = sum W1 over non-degenerate OWMD observables, held-out EVAL seeds",
    "observables_used": USE_OBS,
    "dim_theta": int(DIM),
    "n_iter": N_ITER, "n_train_rollout": N_TRAIN, "n_eval_rollout": N_EVAL,
    "guards": f"box(theta_warm+-{BOX}*scale)+drift_reg(lam{LAM})+gradclip50, NO early stop",
    "d_real_warmstart_eval": d_warm,
    "d_real_min_eval": min_d,
    "d_real_min_at_iter": min_iter,
    "d_real_final_eval": d_final,
    "d_real_best_eval_n80": d_final_hi,
    "best_iter": best_iter,
    "rel_best_dip_vs_warm": float((d_warm - min_d) / (d_warm + 1e-12)),
    "rel_improve_final_vs_warm": float((d_warm - d_final) / (d_warm + 1e-12)),
    "verdict": verdict,
    "curve": curve,
    "elapsed_sec": time.time() - T0,
}
np.save("/root/_mw_hybrid_extended_theta.npy", best_theta)
with open("/root/_mw_hybrid_extended_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"DONE elapsed={time.time()-T0:.0f}s", flush=True)
print(f"  d_warm={d_warm:.4f} d_min={min_d:.4f}@{min_iter} d_final={d_final:.4f} "
      f"verdict={verdict}", flush=True)
