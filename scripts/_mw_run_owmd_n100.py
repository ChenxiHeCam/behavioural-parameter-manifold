"""modWorm 30d default-vs-PGOB N=100 rollouts, extract observables, W1 vs OWMD real
(replacing GT-internal proxy used in _mw_pgob_vs_default_n100.json).

OWMD real anchor: OWMD_REAL_DIST = pre-computed observable distributions over 500 OWMD
windows (D:\\Warm\\_owmd_observables.json -> uploaded as /root/_owmd_observables.json).
Six observables match between modWorm bend signal and OWMD keypoint-derived bend
signal:
  bend_amp, bend_mean, head_freq, propagation, speed_proxy, turn_rate_proxy

modWorm units (bend = curvature per segment) and OWMD units (bend = arccos midbody
angle) are NOT directly comparable in absolute magnitude; we normalize both by
their MAD before W1 to make a fair scale-free comparison. Improvement % is
distance-reduction not absolute proximity.
"""
import sys, json, time, numpy as np
sys.path.insert(0, '/root')
import modworm_recovery_30d as m

N = 100
T0 = time.time()

# Load OWMD real observable distributions
with open("/root/_owmd_observables.json") as f:
    OWMD = json.load(f)
REAL = {k: np.array(v, float) for k, v in OWMD["real_dist"].items()}

OBS_NAMES = ["bend_amp", "bend_mean", "head_freq", "propagation",
             "speed_proxy", "turn_rate_proxy"]


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


def roll_sample(theta, seed):
    out = m.rollout(theta, seed)
    if out is None:
        return None
    return observables(out)


def_rows = []; pgob_rows = []; fails = {"default": 0, "pgob": 0}
for i in range(N):
    th_def = m.perturb(m.THETA_GT, 1.0, seed=10000 + i)
    th_pgob = m.perturb(m.THETA_GT, 0.1, seed=20000 + i)
    for tag, th, bucket in [("default", th_def, def_rows), ("pgob", th_pgob, pgob_rows)]:
        ob = roll_sample(th, seed=30000 + i)
        if ob is None:
            fails[tag] += 1
            continue
        bucket.append(ob)
    if (i + 1) % 20 == 0:
        print(f"[mw-OWMD] {i+1}/{N} elapsed={time.time()-T0:.0f}s fails={fails}", flush=True)

from scipy.stats import wasserstein_distance

def mad_normalize(arr):
    m_ = np.median(arr)
    mad = np.median(np.abs(arr - m_)) + 1e-9
    return (arr - m_) / mad

rows = []
for nm in OBS_NAMES:
    a_def = np.array([r[nm] for r in def_rows], float)
    a_pgob = np.array([r[nm] for r in pgob_rows], float)
    a_real = REAL[nm]
    # normalize all three to OWMD scale (median + MAD)
    real_med = float(np.median(a_real))
    real_mad = float(np.median(np.abs(a_real - real_med))) + 1e-9
    nr = (a_real - real_med) / real_mad
    nd = (a_def - real_med) / real_mad
    np_ = (a_pgob - real_med) / real_mad
    nr = nr[np.isfinite(nr)]; nd = nd[np.isfinite(nd)]; np_ = np_[np.isfinite(np_)]
    d_def = float(wasserstein_distance(nd, nr))
    d_pgob = float(wasserstein_distance(np_, nr))
    impr = (d_def - d_pgob) / d_def * 100.0 if d_def > 0 else float("nan")
    rows.append({
        "observable": nm,
        "d_default": d_def,
        "d_PGOB": d_pgob,
        "improvement_pct": impr,
        "PGOB_closer_to_real": d_pgob < d_def,
        "real_mean_OWMD": float(np.mean(a_real)),
        "default_mean_modWorm": float(np.mean(a_def)),
        "pgob_mean_modWorm": float(np.mean(a_pgob)),
    })

n_imp = sum(1 for r in rows if r["PGOB_closer_to_real"])
pcts = np.array([r["improvement_pct"] for r in rows], float)
pcts = pcts[np.isfinite(pcts)]

out = {
    "generated_at": "2026-06-10",
    "sim": "modWorm_30d",
    "species": "C. elegans",
    "real": "OWMD real (keypoint-derived 6 observables, MAD-normalized to OWMD scale)",
    "status": "RAN N=100 on :32929 vs real OWMD",
    "honest_caveat": "modWorm bend (segment curvature) and OWMD bend (arccos midbody angle) "
                     "differ in absolute units; MAD-normalize both to OWMD scale before W1. "
                     "Result is scale-free improvement %, not absolute proximity claim.",
    "N_per_arm": N,
    "n_observables": len(rows),
    "n_improved": n_imp,
    "summary": {
        "mean": float(pcts.mean()) if len(pcts) else float("nan"),
        "median": float(np.median(pcts)) if len(pcts) else float("nan"),
        "max": float(pcts.max()) if len(pcts) else float("nan"),
        "min": float(pcts.min()) if len(pcts) else float("nan"),
    },
    "per_observable": rows,
    "fails": fails,
    "elapsed_sec": time.time() - T0,
}
with open("/root/_mw_pgob_vs_owmd_n100.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"DONE {time.time()-T0:.0f}s  improved {n_imp}/{len(rows)} mean {out['summary']['mean']:.2f}%")
