"""Compute BAAIWorm T=1000 long-rollout observables and closeness vs OWMD N2.

Input: D:/Warm/long_rollouts/baaiworm/baai_long_sota_cond50_step1000.npy
Shape: (1000, 17, 3) (x, y, z) per body point.

Reuse the T=100/300 observable definitions implicit in claim9_time_extrap_summary.json:
  speed_mean, speed_var, turn_rate.
Reference: OWMD_N2_5010snippets_100f (speed_mean=0.0079775, speed_var=4.5564e-05, turn_rate=0.04338).
"""
import json, numpy as np, os

T100 = "D:/Warm/long_rollouts/baaiworm/baai_long_sota_cond50_step300.npy"  # use first 100/300
T1000 = "D:/Warm/long_rollouts/baaiworm/baai_long_sota_cond50_step1000.npy"
T2000_LOCAL = "D:/Warm/long_rollouts/baaiworm/baai_long_sota_cond50_step2000.npy"

REF_OWMD = {"speed_mean": 0.00797748938202858,
            "speed_var": 4.55640401924029e-05,
            "turn_rate": 0.04337942227721214}


def head_speed(traj):
    # head = body point 0
    head = traj[:, 0, :2]  # x,y
    d = np.diff(head, axis=0)
    sp = np.linalg.norm(d, axis=1)
    return sp


def turn_rate(traj):
    head = traj[:, 0, :2]
    d = np.diff(head, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    da = np.diff(np.unwrap(ang))
    return float(np.mean(np.abs(da)))


def obs(traj):
    sp = head_speed(traj)
    return {"speed_mean": float(np.mean(sp)),
            "speed_var": float(np.var(sp)),
            "turn_rate": turn_rate(traj)}


def closeness(o, ref):
    per_key = {k: abs(o[k] - ref[k]) / (abs(ref[k]) + 1e-12) for k in ref}
    return float(np.mean(list(per_key.values()))), per_key


def main():
    out = {"sims": {"baaiworm": {"lengths": {}}},
           "reference_OWMD_N2": REF_OWMD}

    arr300 = np.load(T100)
    for n in (100, 300):
        sub = arr300[:n]
        o = obs(sub)
        c, pk = closeness(o, REF_OWMD)
        out["sims"]["baaiworm"]["lengths"][str(n)] = {
            "observables": o, "closeness": c, "per_key": pk,
            "n_frames": n, "source": T100}

    arr1k = np.load(T1000)
    print("T=1000 shape:", arr1k.shape)
    o = obs(arr1k)
    c, pk = closeness(o, REF_OWMD)
    out["sims"]["baaiworm"]["lengths"]["1000"] = {
        "observables": o, "closeness": c, "per_key": pk,
        "n_frames": int(arr1k.shape[0]), "source": T1000,
        "ckpt": "full5_merge_sota_combo_ionpass_T8",
        "sim_dt_sec_total": 2452.4}

    if os.path.exists(T2000_LOCAL):
        arr2k = np.load(T2000_LOCAL)
        o = obs(arr2k)
        c, pk = closeness(o, REF_OWMD)
        out["sims"]["baaiworm"]["lengths"]["2000"] = {
            "observables": o, "closeness": c, "per_key": pk,
            "n_frames": int(arr2k.shape[0]), "source": T2000_LOCAL,
            "ckpt": "full5_merge_sota_combo_ionpass_T8"}
    else:
        out["sims"]["baaiworm"]["lengths"]["2000"] = {
            "status": "RUNNING_on_server_37565",
            "launcher": "/root/_baai_T2000_launcher.py",
            "log": "/root/baai_T2000.log",
            "expected_completion_min": "80-160 (CPU contention from concurrent ablation)",
        }

    # Trend table + spearman across testable points
    pts = []
    for k in ("100", "300", "1000"):
        d = out["sims"]["baaiworm"]["lengths"].get(k, {})
        if "closeness" in d:
            pts.append((int(k), d["closeness"]))
    if "closeness" in out["sims"]["baaiworm"]["lengths"].get("2000", {}):
        pts.append((2000, out["sims"]["baaiworm"]["lengths"]["2000"]["closeness"]))

    out["trend"] = {"points": pts}
    if len(pts) >= 3:
        from scipy.stats import spearmanr
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        rho, p = spearmanr(xs, ys)
        out["trend"]["spearman_rho"] = float(rho)
        out["trend"]["spearman_p"] = float(p)
        # also Delta closeness max - min
        out["trend"]["delta_max_min"] = float(max(ys) - min(ys))
        out["trend"]["relative_drift_pct"] = float((max(ys) - min(ys)) / min(ys) * 100)

    out["claim9_verdict"] = {
        "n_testable_lengths": len(pts),
        "max_lengths_tested": max(p[0] for p in pts) if pts else None,
    }
    if len(pts) >= 3:
        rho = out["trend"]["spearman_rho"]
        drift = out["trend"]["relative_drift_pct"]
        if abs(rho) < 0.6 and drift < 30:
            verdict = "STRONG (closeness stable across 3+ time scales; |rho|<0.6 AND drift<30%)"
        elif drift < 50:
            verdict = "MODERATE (some drift but bounded)"
        else:
            verdict = "WEAK (large drift)"
        out["claim9_verdict"]["interpretation"] = verdict
        out["claim9_verdict"]["closeness_stable_across_T"] = bool(abs(rho) < 0.6 and drift < 30)

    out_path = "D:/Warm/claim9_baai_T1000_T2000_real.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("WROTE", out_path)
    print(json.dumps({"trend": out.get("trend"), "verdict": out.get("claim9_verdict")},
                     indent=2))


if __name__ == "__main__":
    main()
