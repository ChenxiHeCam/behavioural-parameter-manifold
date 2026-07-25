"""modWorm: 12 observable + perturb sweep + paired bootstrap.

Extends _mw_simagnostic_run.py with:
  - 6 NEW observables: head_bend_amp, tail_bend_amp, undulation_phase_std,
    midbody_curv_cv, speed_rev_freq, pirouette_rate (all sim-agnostic, from
    body-frame curvature (T, K-1)).
  - perturb sweep: PGOB scale in {0.02, 0.05, 0.10, 0.15, 0.30, 0.50}.
  - paired bootstrap CI95 per observable per scale.

Reuses OWMD pipeline from _mw_simagnostic_run.py for real side.
"""
import sys, os, json, time, glob
import numpy as np
sys.path.insert(0, "/root")
import modworm_recovery_30d as m
import h5py
from scipy.stats import wasserstein_distance

N = 60                                  # rollouts per arm per scale (was 100; 6 scales)
N_OWMD = 200
K = 20
T_OWMD = 200
SCALES = [0.02, 0.05, 0.10, 0.15, 0.30, 0.50]
T0 = time.time()
RNG = np.random.RandomState(42)

# --------------------------- OWMD reuse (verbatim) ---------------------------

def skeleton_to_curvature(skel):
    T, P, _ = skel.shape
    out = np.zeros((T, K, 2), dtype=np.float64)
    last_good = None; n_bad = 0
    for t in range(T):
        pts = skel[t]
        if not np.all(np.isfinite(pts)):
            n_bad += 1
            if last_good is None: continue
            out[t] = last_good; continue
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(d)])
        if s[-1] <= 1e-9:
            n_bad += 1
            if last_good is None: continue
            out[t] = last_good; continue
        u = s / s[-1]; tgt = np.linspace(0, 1, K)
        frame = np.zeros((K, 2))
        for ax in range(2):
            frame[:, ax] = np.interp(tgt, u, pts[:, ax])
        out[t] = frame; last_good = frame
    if last_good is None or n_bad > T // 3: return None
    if not np.all(np.isfinite(out)): return None
    seg = np.diff(out, axis=1)
    ang = np.arctan2(seg[..., 1], seg[..., 0])
    ang = np.unwrap(ang, axis=1)
    ang = ang - ang.mean(axis=1, keepdims=True)
    return ang


def owmd_windows(n_target=N_OWMD):
    files = sorted(glob.glob("/root/owmd_mut/*.hdf5"))
    wins = []; fi = 0
    while len(wins) < n_target and fi < len(files):
        fp = files[fi]; fi += 1
        try:
            with h5py.File(fp, "r") as f:
                sk = f["coordinates/skeletons"][:]
        except Exception: continue
        T = sk.shape[0]
        if T < T_OWMD + 100: continue
        for k in range(40):
            if len(wins) >= n_target: break
            st = 100 + k * T_OWMD
            if st + T_OWMD > T: break
            cur = skeleton_to_curvature(sk[st:st + T_OWMD])
            if cur is None or not np.all(np.isfinite(cur)): continue
            wins.append(cur)
    return wins


def bend_to_curvature(bend):
    T, n = bend.shape
    src = np.linspace(0, 1, n); tgt = np.linspace(0, 1, K - 1)
    out = np.zeros((T, K - 1))
    for t in range(T):
        out[t] = np.interp(tgt, src, bend[t])
    out = out - out.mean(axis=1, keepdims=True)
    return out


# --------------------------- 12 observables ---------------------------

def observables12(curv):
    """curv: (T, K-1) -> 12 sim-agnostic observables."""
    T, Km1 = curv.shape
    X = curv - curv.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    var = (S ** 2) / max(1e-12, (S ** 2).sum())
    coef = U * S
    eig1 = coef[:, 0]
    e = eig1 - eig1.mean()
    if e.std() < 1e-9: bf = 0.0
    else:
        f = np.abs(np.fft.rfft(e))
        bf = float((f[1:].argmax() + 1) / len(e))
    de1 = np.abs(np.diff(eig1))
    af = float((de1 > (np.median(de1) + 1e-12)).mean())
    pc = float(np.mean(np.abs(np.diff(curv, axis=0))))
    # NEW 6
    head_idx = slice(0, max(1, Km1 // 3))
    tail_idx = slice(Km1 - max(1, Km1 // 3), Km1)
    mid_idx = slice(Km1 // 3, 2 * Km1 // 3)
    head_bend_amp = float(np.mean(np.abs(curv[:, head_idx])))
    tail_bend_amp = float(np.mean(np.abs(curv[:, tail_idx])))
    # undulation phase std: phase of eig2 vs eig1 (Stephens style)
    if coef.shape[1] >= 2:
        eig2 = coef[:, 1]
        ph = np.arctan2(eig2, eig1)
        # circular std
        sin_m = float(np.mean(np.sin(ph))); cos_m = float(np.mean(np.cos(ph)))
        R = np.sqrt(sin_m**2 + cos_m**2)
        und_ph_std = float(np.sqrt(max(0.0, -2*np.log(max(1e-9, R)))))
    else:
        und_ph_std = 0.0
    mid_series = np.mean(np.abs(curv[:, mid_idx]), axis=1)
    mid_curv_cv = float(mid_series.std() / max(1e-9, abs(mid_series.mean())))
    # speed reversal: sign flips of d(eig1)
    de1_signed = np.diff(eig1)
    sg = np.sign(de1_signed)
    sg = sg[sg != 0]
    if len(sg) > 1:
        rev_freq = float((sg[1:] != sg[:-1]).mean())
    else:
        rev_freq = 0.0
    # pirouette: fraction of windows where |d(eig1)| > 3*median (sharp turns)
    med = np.median(de1)
    pir_rate = float((de1 > 3 * med).mean())
    return {
        "eig1_var": float(var[0]),
        "eig2_var": float(var[1]) if len(var) > 1 else 0.0,
        "eig3_var": float(var[2]) if len(var) > 2 else 0.0,
        "bend_freq": bf,
        "active_frac": af,
        "path_curv": pc,
        "head_bend_amp": head_bend_amp,
        "tail_bend_amp": tail_bend_amp,
        "undulation_phase_std": und_ph_std,
        "midbody_curv_cv": mid_curv_cv,
        "speed_rev_freq": rev_freq,
        "pirouette_rate": pir_rate,
    }


# --------------------------- run ---------------------------
print(f"[mw12] loading OWMD N={N_OWMD}", flush=True)
owmd_curvs = owmd_windows(N_OWMD)
real_rows = [observables12(c) for c in owmd_curvs]
real_rows = [r for r in real_rows if r is not None]
print(f"[mw12] OWMD obs ready: {len(real_rows)}", flush=True)

# default: scale=1.0 (full random perturb) — match original
def_rows = []; fails_def = 0
for i in range(N):
    th = m.perturb(m.THETA_GT, 1.0, seed=10000 + i)
    bend = m.rollout(th, seed=30000 + i)
    if bend is None or not np.all(np.isfinite(bend)): fails_def += 1; continue
    ob = observables12(bend_to_curvature(bend))
    if ob is None: fails_def += 1; continue
    def_rows.append(ob)
print(f"[mw12] default N={len(def_rows)} fails={fails_def}", flush=True)

# PGOB sweep
sweep_results = {}
for sc in SCALES:
    pg_rows = []; fails_pg = 0
    for i in range(N):
        th = m.perturb(m.THETA_GT, sc, seed=20000 + i + int(sc * 1e5))
        bend = m.rollout(th, seed=40000 + i + int(sc * 1e5))
        if bend is None or not np.all(np.isfinite(bend)): fails_pg += 1; continue
        ob = observables12(bend_to_curvature(bend))
        if ob is None: fails_pg += 1; continue
        pg_rows.append(ob)
    sweep_results[sc] = {"rows": pg_rows, "fails": fails_pg}
    print(f"[mw12] scale={sc} N={len(pg_rows)} fails={fails_pg} el={time.time()-T0:.0f}s", flush=True)

NAMES = ["eig1_var","eig2_var","eig3_var","bend_freq","active_frac","path_curv",
         "head_bend_amp","tail_bend_amp","undulation_phase_std","midbody_curv_cv",
         "speed_rev_freq","pirouette_rate"]

def w1_normalized(ar, ad):
    rm = float(np.median(ar))
    rmd = float(np.median(np.abs(ar - rm)))
    if rmd < 1e-6: rmd = float(ar.std())
    if rmd < 1e-6:
        combined = np.concatenate([ar, ad])
        rmd = max(float(np.abs(combined - rm).max()), 1e-6)
    return float(wasserstein_distance((ad - rm) / rmd, (ar - rm) / rmd))

def paired_bootstrap_ci(ar, ad, ap, B=1000, seed=7):
    """Paired bootstrap on (d_def - d_pgob) for improvement CI."""
    rng = np.random.RandomState(seed)
    diffs = []
    n_d = len(ad); n_p = len(ap)
    for _ in range(B):
        i_d = rng.randint(0, n_d, size=n_d)
        i_p = rng.randint(0, n_p, size=n_p)
        # bootstrap real too (independent)
        i_r = rng.randint(0, len(ar), size=len(ar))
        ad_b = ad[i_d]; ap_b = ap[i_p]; ar_b = ar[i_r]
        d_d = w1_normalized(ar_b, ad_b); d_p = w1_normalized(ar_b, ap_b)
        if d_d > 0:
            diffs.append((d_d - d_p) / d_d * 100.0)
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(diffs.mean())

# Build results
per_scale = {}
for sc in SCALES:
    pg_rows = sweep_results[sc]["rows"]
    rows = []
    for nm in NAMES:
        ar = np.array([r[nm] for r in real_rows], float); ar = ar[np.isfinite(ar)]
        ad = np.array([r[nm] for r in def_rows], float); ad = ad[np.isfinite(ad)]
        ap = np.array([r[nm] for r in pg_rows], float); ap = ap[np.isfinite(ap)]
        if len(ar) < 3 or len(ad) < 3 or len(ap) < 3:
            rows.append({"observable": nm, "skip": True}); continue
        d_def = w1_normalized(ar, ad); d_pgb = w1_normalized(ar, ap)
        impr = (d_def - d_pgb) / d_def * 100.0 if d_def > 0 else float("nan")
        ci_lo, ci_hi, boot_mean = paired_bootstrap_ci(ar, ad, ap, B=500, seed=int(sc*1e5)+7)
        # Cohen's d on per-rollout observable values (default vs pgob)
        pooled_sd = np.sqrt(0.5 * (ad.var() + ap.var()))
        cohen_d = float((ad.mean() - ap.mean()) / pooled_sd) if pooled_sd > 1e-9 else 0.0
        rows.append({
            "observable": nm,
            "d_default": d_def, "d_PGOB": d_pgb,
            "improvement_pct": impr,
            "improvement_ci95_low": ci_lo,
            "improvement_ci95_high": ci_hi,
            "boot_mean_improvement_pct": boot_mean,
            "cohen_d": cohen_d,
            "PGOB_closer_to_real": d_pgb < d_def,
            "real_mean_OWMD": float(np.mean(ar)),
            "default_mean_modWorm": float(np.mean(ad)),
            "pgob_mean_modWorm": float(np.mean(ap)),
        })
    scored = [r for r in rows if not r.get("skip")]
    n_imp = sum(1 for r in scored if r["PGOB_closer_to_real"])
    pcts = np.array([r["improvement_pct"] for r in scored], float)
    pcts = pcts[np.isfinite(pcts)]
    per_scale[str(sc)] = {
        "scale": sc,
        "N_pgob": len(pg_rows),
        "fails_pgob": sweep_results[sc]["fails"],
        "summary": {
            "mean": float(pcts.mean()) if len(pcts) else float("nan"),
            "median": float(np.median(pcts)) if len(pcts) else float("nan"),
            "max": float(pcts.max()) if len(pcts) else float("nan"),
            "min": float(pcts.min()) if len(pcts) else float("nan"),
            "n_obs": len(scored), "n_improved": n_imp,
        },
        "per_observable": rows,
    }

# best scale by mean improvement
best_sc = max(per_scale.keys(), key=lambda k: per_scale[k]["summary"]["mean"]
              if np.isfinite(per_scale[k]["summary"]["mean"]) else -1e9)
out = {
    "generated_at": "2026-06-10",
    "sim": "modWorm_30d",
    "experiment": "12 observable + perturb sweep + paired bootstrap CI95",
    "N_per_arm": N, "N_OWMD_windows": len(real_rows),
    "scales_tested": SCALES,
    "best_scale": float(best_sc),
    "best_summary": per_scale[best_sc]["summary"],
    "per_scale": per_scale,
    "default_N": len(def_rows), "default_fails": fails_def,
    "elapsed_sec": time.time() - T0,
}
with open("/root/_mw_better_attempt.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"DONE el={time.time()-T0:.0f}s best_scale={best_sc} mean={per_scale[best_sc]['summary']['mean']:.2f}%")
