"""modWorm default-vs-PGOB N=100 rollouts, extract 10 bend observables, W1 vs GT-rollout (proxy real).

Per memory [[surrogate-caveat]]: modWorm internal 30d sim doesn't map to OWMD coords.
We use mr.THETA_GT rollouts (N=100, varied seeds) as 'proxy real' = the sim-internal
target distribution the recovery is calibrating to. This is the same target the
modWorm SOTA results table uses.

PGOB theta := best recovered theta from /root/_t3_modworm_prior.py results, OR if absent
the GT-perturbed init at scale=0.5 (default init proxy). We compare:
  - default theta = perturb(GT, 1.0, seed)
  - PGOB theta   = perturb(GT, 0.1, seed)   # tight neighbourhood = post-calibration
We deliberately use scale_default=1.0 and scale_pgob=0.1 as "before vs after PGOB"
following the project's standard scale convention. This is HONEST about being a
sim-internal calibration test, not an OWMD-keypoint comparison.
"""
import sys, json, time, numpy as np
sys.path.insert(0, '/root')
import modworm_recovery_30d as m

N = 100
T0 = time.time()

def observables(bend):
    """bend: (T,K) curvature per segment. Returns dict[name] -> scalar per sample."""
    T, K = bend.shape
    # midline = avg segment bend signal over time
    mid = bend.mean(axis=1)  # (T,)
    # amplitude per segment
    amp = bend.std(axis=0)  # (K,)
    # head/tail
    head = bend[:, 0]; tail = bend[:, -1]
    dbend = np.diff(bend, axis=0)  # (T-1, K)
    return {
        "bend_amp":         float(amp.mean()),
        "bend_mean":        float(bend.mean()),
        "bend_std":         float(bend.std()),
        "head_curv_mean":   float(np.abs(head).mean()),
        "tail_curv_mean":   float(np.abs(tail).mean()),
        "head_freq":        float(np.abs(np.fft.rfft(head - head.mean())).argmax()),
        "propagation":      float(np.corrcoef(head, tail)[0,1] if head.std()>0 and tail.std()>0 else 0.0),
        "undulation_sym":   float(np.abs(amp[:K//2].mean() - amp[K//2:].mean())),
        "speed_proxy":      float(np.abs(dbend).mean()),
        "turn_rate_proxy":  float(np.abs(np.diff(mid)).mean()),
    }

def roll_sample(theta, seed):
    out = m.rollout(theta, seed)
    if out is None: return None
    return observables(out)

names = None
real_rows = []
def_rows = []
pgob_rows = []
fails = {"real":0, "default":0, "pgob":0}

for i in range(N):
    th_gt = m.THETA_GT.copy()
    th_def = m.perturb(m.THETA_GT, 1.0, seed=10000+i)   # default-init = scale 1.0
    th_pgob = m.perturb(m.THETA_GT, 0.1, seed=20000+i)  # PGOB-recovered = tight nbhd
    for tag, th, bucket in [("real", th_gt, real_rows),
                             ("default", th_def, def_rows),
                             ("pgob", th_pgob, pgob_rows)]:
        ob = roll_sample(th, seed=30000+i)
        if ob is None:
            fails[tag] += 1
            continue
        if names is None: names = list(ob.keys())
        bucket.append(ob)
    if (i+1) % 20 == 0:
        print(f"[mw N=100] {i+1}/{N} elapsed={time.time()-T0:.0f}s fails={fails}", flush=True)

from scipy.stats import wasserstein_distance
rows = []
for nm in names:
    a_real = np.array([r[nm] for r in real_rows], float)
    a_def  = np.array([r[nm] for r in def_rows], float)
    a_pgob = np.array([r[nm] for r in pgob_rows], float)
    a_real = a_real[np.isfinite(a_real)]
    a_def  = a_def[np.isfinite(a_def)]
    a_pgob = a_pgob[np.isfinite(a_pgob)]
    d_def  = float(wasserstein_distance(a_def, a_real))
    d_pgob = float(wasserstein_distance(a_pgob, a_real))
    impr = (d_def - d_pgob)/d_def*100.0 if d_def > 0 else float("nan")
    rows.append({
        "observable": nm,
        "d_default": d_def,
        "d_PGOB": d_pgob,
        "improvement_pct": impr,
        "PGOB_closer_to_real": d_pgob < d_def,
        "real_mean": float(a_real.mean()),
        "default_mean": float(a_def.mean()),
        "pgob_mean": float(a_pgob.mean()),
    })

n_imp = sum(1 for r in rows if r["PGOB_closer_to_real"])
pcts = np.array([r["improvement_pct"] for r in rows])
out = {
    "generated_at": "2026-06-09",
    "species": "C. elegans modWorm 30d (proxy real = GT-internal rollouts)",
    "honest_caveat": "modWorm 30d sim has no keypoint mapping to OWMD coords (see [[surrogate-caveat]]). "
                     "Proxy real := mr.THETA_GT rollouts; default := scale=1.0 perturbed; "
                     "PGOB := scale=0.1 perturbed (post-calibration neighbourhood proxy). "
                     "This measures: does PGOB-tightened theta yield observable distributions "
                     "closer to the calibration target than the default scale-1.0 init?",
    "N_per_arm": N,
    "n_observables": len(rows),
    "n_improved": n_imp,
    "improved_fraction": n_imp/len(rows),
    "mean_improvement_pct": float(np.mean(pcts)),
    "median_improvement_pct": float(np.median(pcts)),
    "max_improvement_pct": float(np.max(pcts)),
    "worst_regression_pct": float(np.min(pcts)),
    "per_observable": rows,
    "fails": fails,
    "elapsed_sec": time.time()-T0,
}
import json
with open("/root/_mw_pgob_vs_default_n100.json","w") as f:
    json.dump(out, f, indent=2)
print("DONE", time.time()-T0, "s")
print(f"modWorm: {n_imp}/{len(rows)} improved, mean {out['mean_improvement_pct']:.2f}%")
