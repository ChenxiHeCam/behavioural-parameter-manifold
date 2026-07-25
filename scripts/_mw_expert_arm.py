"""modWorm THREE-arm three-segment run: random(scale=1.0) / PGOB(scale=0.1) /
expert(scale=0.0 == THETA_GT). Reuses the EXACT sim-agnostic eigenworm + dynamics
observables and the EXACT OWMD-derived W1 normalization from _mw_simagnostic_run.py
so d_random / d_pgob / d_expert are unit-matched and directly comparable to the
stored pgob_vs_default_4sim.json modWorm cell.

The only addition vs the original is a third arm: expert = perturb(THETA_GT, 0.0).
W1 normalization (median + MAD) is computed from the OWMD real distribution ONLY,
identical to the original, hence arm-independent and comparable across all arms.
"""
import sys, os, json, time, glob
import numpy as np
sys.path.insert(0, "/root")
import modworm_recovery_30d as m
import h5py
from scipy.stats import wasserstein_distance

N = 100
N_OWMD = 200
K = 20
T_OWMD = 200
T0 = time.time()


def skeleton_to_curvature(skel):
    T, P, _ = skel.shape
    out = np.zeros((T, K, 2), dtype=np.float64)
    last_good = None
    n_bad = 0
    for t in range(T):
        pts = skel[t]
        if not np.all(np.isfinite(pts)):
            n_bad += 1
            if last_good is None:
                continue
            out[t] = last_good
            continue
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(d)])
        if s[-1] <= 1e-9:
            n_bad += 1
            if last_good is None:
                continue
            out[t] = last_good
            continue
        u = s / s[-1]
        tgt = np.linspace(0, 1, K)
        frame = np.zeros((K, 2))
        for ax in range(2):
            frame[:, ax] = np.interp(tgt, u, pts[:, ax])
        out[t] = frame
        last_good = frame
    if last_good is None or n_bad > T // 3:
        return None
    if not np.all(np.isfinite(out)):
        return None
    seg = np.diff(out, axis=1)
    ang = np.arctan2(seg[..., 1], seg[..., 0])
    ang = np.unwrap(ang, axis=1)
    ang = ang - ang.mean(axis=1, keepdims=True)
    return ang


def owmd_windows(n_target=N_OWMD):
    files = sorted(glob.glob("/root/owmd_mut/*.hdf5"))
    wins = []
    fi = 0
    while len(wins) < n_target and fi < len(files):
        fp = files[fi]; fi += 1
        try:
            with h5py.File(fp, "r") as f:
                sk = f["coordinates/skeletons"][:]
        except Exception:
            continue
        T = sk.shape[0]
        if T < T_OWMD + 100:
            continue
        for k in range(40):
            if len(wins) >= n_target:
                break
            st = 100 + k * T_OWMD
            if st + T_OWMD > T:
                break
            w = sk[st:st + T_OWMD]
            cur = skeleton_to_curvature(w)
            if cur is None or not np.all(np.isfinite(cur)):
                continue
            wins.append(cur)
    return wins


def bend_to_curvature(bend):
    T, n = bend.shape
    src = np.linspace(0, 1, n)
    tgt = np.linspace(0, 1, K - 1)
    out = np.zeros((T, K - 1))
    for t in range(T):
        out[t] = np.interp(tgt, src, bend[t])
    out = out - out.mean(axis=1, keepdims=True)
    return out


def observables(curv, n_modes=3):
    X = curv - curv.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    var = (S ** 2) / max(1e-12, (S ** 2).sum())
    coef = U * S
    eig1 = coef[:, 0]
    e = eig1 - eig1.mean()
    if e.std() < 1e-9:
        bf = 0.0
    else:
        f = np.abs(np.fft.rfft(e))
        bf = float((f[1:].argmax() + 1) / len(e))
    de1 = np.abs(np.diff(eig1))
    af = float((de1 > (np.median(de1) + 1e-12)).mean())
    pc = float(np.mean(np.abs(np.diff(curv, axis=0))))
    return {
        "eig1_var": float(var[0]),
        "eig2_var": float(var[1]) if len(var) > 1 else 0.0,
        "eig3_var": float(var[2]) if len(var) > 2 else 0.0,
        "bend_freq": bf,
        "active_frac": af,
        "path_curv": pc,
    }


print(f"[mw3] loading OWMD windows N={N_OWMD}", flush=True)
owmd_curvs = owmd_windows(N_OWMD)
real_rows = [observables(c) for c in owmd_curvs]
real_rows = [r for r in real_rows if r is not None]
print(f"[mw3] OWMD observables: {len(real_rows)} elapsed={time.time()-T0:.0f}s", flush=True)

# three arms; SAME per-traj seeds as original (default=10000+i scale1.0,
# pgob=20000+i scale0.1) + new expert=THETA_GT (scale 0.0). rollout seed 30000+i
# identical to original so default/pgob arms reproduce stored numbers exactly.
arms = {"random": [], "pgob": [], "expert": []}
fails = {"random": 0, "pgob": 0, "expert": 0}
for i in range(N):
    th_rand = m.perturb(m.THETA_GT, 1.0, seed=10000 + i)
    th_pgob = m.perturb(m.THETA_GT, 0.1, seed=20000 + i)
    th_exp = m.THETA_GT  # scale 0.0
    for tag, th in [("random", th_rand), ("pgob", th_pgob), ("expert", th_exp)]:
        bend = m.rollout(th, seed=30000 + i)
        if bend is None or not np.all(np.isfinite(bend)):
            fails[tag] += 1; continue
        ob = observables(bend_to_curvature(bend))
        if ob is None:
            fails[tag] += 1; continue
        arms[tag].append(ob)
    if (i + 1) % 20 == 0:
        print(f"[mw3] {i+1}/{N} elapsed={time.time()-T0:.0f}s fails={fails}", flush=True)

NAMES = ["eig1_var", "eig2_var", "eig3_var", "bend_freq", "active_frac", "path_curv"]
rows = []
for nm in NAMES:
    ar = np.array([r[nm] for r in real_rows], float)
    ar = ar[np.isfinite(ar)]
    a = {k: np.array([r[nm] for r in v], float) for k, v in arms.items()}
    a = {k: x[np.isfinite(x)] for k, x in a.items()}
    if len(ar) < 3 or any(len(x) < 3 for x in a.values()):
        rows.append({"observable": nm, "skip": True}); continue
    rm = float(np.median(ar))
    rmd_raw = float(np.median(np.abs(ar - rm)))
    if rmd_raw < 1e-6:
        rmd_raw = float(ar.std())
    if rmd_raw < 1e-6:
        combined = np.concatenate([ar] + list(a.values()))
        rmd_raw = max(float(np.abs(combined - rm).max()), 1e-6)
    rmd = rmd_raw
    nr = (ar - rm) / rmd
    d = {k: float(wasserstein_distance((x - rm) / rmd, nr)) for k, x in a.items()}
    rows.append({
        "observable": nm,
        "d_random": d["random"],
        "d_pgob": d["pgob"],
        "d_expert": d["expert"],
        "real_mean_OWMD": float(np.mean(ar)),
        "random_mean": float(np.mean(a["random"])),
        "pgob_mean": float(np.mean(a["pgob"])),
        "expert_mean": float(np.mean(a["expert"])),
    })

scored = [r for r in rows if not r.get("skip")]
agg = {k: float(sum(r["d_" + k] for r in scored)) for k in ["random", "pgob", "expert"]}
out = {
    "sim": "modWorm_30d",
    "generated_at": time.strftime("%Y-%m-%d"),
    "N_per_arm": N, "N_OWMD_windows": len(real_rows),
    "arms": "random=perturb(THETA_GT,1.0)  pgob=perturb(.,0.1)  expert=THETA_GT(scale0.0)",
    "per_observable": rows,
    "fails": fails,
    "aggregate_W1": agg,
    "elapsed_sec": time.time() - T0,
}
with open("/root/_mw_expert_arm_result.json", "w") as f:
    json.dump(out, f, indent=2)
print("DONE", json.dumps(agg), flush=True)
print(f"elapsed={time.time()-T0:.0f}s", flush=True)
