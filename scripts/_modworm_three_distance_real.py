"""GENUINE three-distance for modWorm using the EXACT robust MAD-z formula from
_t1_t3_robust.py:  z = (vec - real_median)/(1.4826*real_MAD); dist = L2 over active obs.

Observable family: the 6 keypoint-derived OWMD W1 observables
  (bend_amp, bend_mean, head_freq, propagation, speed_proxy, turn_rate_proxy).
This is the family in which the modWorm PGOB-recovered (pgob) and deposited-default
arm observable MEANS were genuinely rolled out (N=100 on :32929) and stored in
_mw_pgob_vs_owmd_n100.json, AND in which the real OWMD per-window distribution
(N=500) is stored in _owmd_observables.json. Both arms and the real distribution
are MAD-normalized to OWMD scale (modWorm segment-curvature bend vs OWMD arccos
midbody-angle bend differ in absolute units -> scale-free, documented caveat).

Arms available as raw observable means:
  pgob (recovered) : pgob_mean_modWorm   -> recovered_to_real
  default          : default_mean_modWorm-> default_to_real
Real distribution  : _owmd_observables.json real_dist (500 windows) -> median/MAD/real_to_real
NOT available as raw means in this family: a separate sim-self target rollout
(recovered_to_sim), and PGOB-real / PGOB-hybrid modWorm arms (the real-mode SPSA
in _genuine_modworm_3mode.json did not move: train_loss flat ~3.54e8, dominated by
the head_freq sentinel -> no meaningful real/hybrid modWorm arm to score).
"""
import json, numpy as np

OBS = ["bend_amp","bend_mean","head_freq","propagation","speed_proxy","turn_rate_proxy"]

owmd = json.load(open("D:/Warm/_owmd_observables.json"))
mw   = json.load(open("D:/Warm/_mw_pgob_vs_owmd_n100.json"))

# real per-window matrix (500, 6)
real = np.array([owmd["real_dist"][k] for k in OBS], float).T
print("real matrix", real.shape)

# arm observable mean vectors (bend family, MAD-normalized to OWMD scale)
pm = {r["observable"]: r for r in mw["per_observable"]}
v_pgob    = np.array([pm[k]["pgob_mean_modWorm"]    for k in OBS], float)
v_default = np.array([pm[k]["default_mean_modWorm"] for k in OBS], float)
real_mean_owmd = np.array([pm[k]["real_mean_OWMD"]  for k in OBS], float)  # sanity vs owmd real_means

# robust stats on real distribution
real_median = np.median(real, axis=0)
real_std    = real.std(axis=0, ddof=1)
mad         = np.median(np.abs(real - real_median[None,:]), axis=0)
mad_scaled  = mad*1.4826
mad_safe    = np.where(mad_scaled<1e-12, np.maximum(real_std,1e-12), mad_scaled)
active      = real_std > 1e-9

def zdist(vec):
    z = (vec[active]-real_median[active])/mad_safe[active]
    return float(np.linalg.norm(z))

def real_to_real():
    ds=[]
    for row in real:
        z=(row[active]-real_median[active])/mad_safe[active]
        ds.append(np.linalg.norm(z))
    return float(np.mean(ds))

d_recovered_to_real = zdist(v_pgob)
d_default_to_real   = zdist(v_default)
d_real_to_real      = real_to_real()

act_names = [OBS[i] for i in range(len(OBS)) if active[i]]
per_metric = {}
for i,k in enumerate(OBS):
    if not active[i]: continue
    per_metric[k] = {
        "pgob_recovered_mean": float(v_pgob[i]),
        "default_mean": float(v_default[i]),
        "real_median": float(real_median[i]),
        "real_mad_scaled": float(mad_safe[i]),
        "z_recovered_to_real": float((v_pgob[i]-real_median[i])/mad_safe[i]),
        "z_default_to_real": float((v_default[i]-real_median[i])/mad_safe[i]),
    }

out = {
  "sim":"modWorm_30d",
  "species":"C. elegans",
  "metric":"robust MAD z (identical formula to D:/Warm/_t1_t3_robust.py line 53): L2 over active obs of (arm_mean_vec - real_median)/(1.4826*real_MAD)",
  "observable_family":"6 OWMD keypoint-derived W1 observables (bend_amp,bend_mean,head_freq,propagation,speed_proxy,turn_rate_proxy), MAD-normalized to OWMD scale",
  "real_reference":"_owmd_observables.json real_dist (500 OWMD real windows from real_tensor.npz X_raw) -> median/MAD/real_to_real",
  "arms_source":"_mw_pgob_vs_owmd_n100.json (RAN N=100 on :32929: pgob_mean_modWorm, default_mean_modWorm)",
  "n_real": int(real.shape[0]),
  "n_per_arm": int(mw["N_per_arm"]),
  "active_metrics": act_names,
  "distances": {
    "recovered_to_real": d_recovered_to_real,
    "default_to_real": d_default_to_real,
    "real_to_real": d_real_to_real,
    "recovered_to_sim": None,
    "pgob_real_to_real": None,
    "hybrid_to_real": None,
  },
  "unavailable": {
    "recovered_to_sim":"no modWorm sim-self target rollout stored as raw bend-family observable means",
    "pgob_real_to_real / hybrid_to_real":"modWorm real-mode SPSA did not move (train_loss flat ~3.54e8, head_freq sentinel-dominated in _genuine_modworm_3mode.json); no meaningful PGOB-real/hybrid modWorm arm to score",
  },
  "mode_matched_reading": {
    "recovered_closer_to_real_than_default": bool(d_recovered_to_real < d_default_to_real),
    "recovered_within_3x_real_spread": bool(d_recovered_to_real < 3*d_real_to_real),
  },
  "provenance":"ALL from saved data (no fresh simulation). bend-unit caveat: modWorm and OWMD bend definitions differ; both MAD-normalized to OWMD scale, so distances are scale-free proximities not absolute metric units.",
  "caveat_unit": mw["honest_caveat"],
  "per_metric": per_metric,
}
json.dump(out, open("D:/Warm/modworm_three_distance_real.json","w"), indent=2)
print(json.dumps(out["distances"], indent=2))
print("active:", act_names)
print("SAVED D:/Warm/modworm_three_distance_real.json")
