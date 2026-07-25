"""After /root/baaiworm_ablation_claim4.json finishes on :37565, pull it,
merge into D:\\Warm\\modeling_ablation_summary.json, and refresh
D:\\Warm\\modeling_ablation_curves.png with 3-sim panel."""
import json, os, sys
sys.path.insert(0, r"D:\Warm")
from _ssh_helper import connect

LOCAL_RAW = r"D:\Warm\baaiworm_ablation_claim4.json"
SUMMARY = r"D:\Warm\modeling_ablation_summary.json"
PLOT = r"D:\Warm\modeling_ablation_curves.png"


def pull():
    c = connect("37565")
    sftp = c.open_sftp()
    sftp.get("/root/baaiworm_ablation_claim4.json", LOCAL_RAW)
    sftp.close(); c.close()
    print("pulled ->", LOCAL_RAW)


def aggregate():
    import numpy as np
    with open(LOCAL_RAW) as f:
        raw = json.load(f)
    rows = []
    by_level = {}
    for r in raw["results"]:
        if not r.get("ok"):
            continue
        by_level.setdefault(r["level"], []).append(r)
    for L in sorted(by_level):
        cells = by_level[L]
        Lf = [c["L_final"] for c in cells]
        # observable closeness mean over keys
        cl_means = []
        for c in cells:
            vals = [v for v in c.get("closeness", {}).values()
                    if v == v and v is not None]
            if vals:
                cl_means.append(float(np.mean(vals)))
        rows.append({
            "level": L,
            "L_final_mean": float(np.mean(Lf)),
            "L_final_std": float(np.std(Lf)),
            "obs_closeness_mean": float(np.mean(cl_means)) if cl_means else float("nan"),
            "obs_closeness_std": float(np.std(cl_means)) if cl_means else 0.0,
            "n_seeds": len(cells),
        })
    return rows


def update_summary(baai_rows):
    with open(SUMMARY) as f:
        S = json.load(f)
    S["baaiworm"] = baai_rows
    S["design"]["sims"].append("BAAIWorm (NeuronXCore closed-loop, 5-D)")
    S["notes"]["BAAIWorm_status"] = (
        "Completed 2026-06-09 after :37565 rawmse workers released. "
        "Weight-space corruption (sim is C++ black-box; no per-step state access). "
        "SPSA 10 iter, 1 cond (food50), 5-D theta scale."
    )
    with open(SUMMARY, "w") as f:
        json.dump(S, f, indent=2)
    print("updated", SUMMARY)


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, sim_key, title in [
        (axs[0], "modworm", "modWorm (Cook ODE, 7-D)"),
        (axs[1], "flygym", "FlyGym (MuJoCo, 126-D)"),
        (axs[2], "baaiworm", "BAAIWorm (NeuronXCore, 5-D)"),
    ]:
        rows = summary.get(sim_key, [])
        if not rows:
            ax.set_title(title + " (missing)")
            continue
        xs = [r["level"] for r in rows]
        ys = [r["obs_closeness_mean"] for r in rows]
        es = [r.get("obs_closeness_std", 0.0) for r in rows]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(["clean", "+5%", "+20%", "+50%", "shuf30%", "rand"],
                           rotation=20)
        ax.set_title(title)
        ax.set_xlabel("modeling corruption level")
        ax.axhline(0.5, color="grey", linestyle="--", lw=0.6)
        ax.set_ylim(-0.05, 1.05)
    axs[0].set_ylabel("observable closeness (recovered vs target)")
    fig.suptitle("Claim 4: PGOB needs a faithful underlying simulator")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=140)
    print("wrote", PLOT)


def main():
    pull()
    rows = aggregate()
    print("BAAIWorm rows:", json.dumps(rows, indent=2))
    update_summary(rows)
    with open(SUMMARY) as f:
        S = json.load(f)
    plot(S)


if __name__ == "__main__":
    main()
