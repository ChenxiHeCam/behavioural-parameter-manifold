"""Build Fig: eigenworm identity + complementary tiling + cross-species union growth.
Reads local paper2 JSONs, writes fig_p2_tiling_crossspecies.png into NEXT_PAPER_manifold_subspace/."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = r"D:/Warm/NEXT_PAPER_manifold_subspace"
def L(n):
    with open(os.path.join(D, n)) as f: return json.load(f)

ew   = L("paper2_EW_eigenworm.json")
chem = L("paper2_BAAI_chemo.json")
sat  = L("paper2_SAT_saturation.json")
rod  = L("paper2_RODENT_manifold.json")
lar  = L("paper2_LARVA_manifold.json")
mb   = L("paper2_MB_behaviour_specificity.json")

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))

# (a) eigenworm cumulative variance (real OWMD) + model alignment
real_cv = ew["real_varexp_top4"]
ax[0].plot(range(1, 5), real_cv, "o-", color="#1b3a6b", lw=2, ms=7, label="real OWMD N2")
ax[0].axhline(0.9, ls=":", color="grey", lw=1)
ax[0].set_xticks(range(1, 5))
ax[0].set_ylim(0, 1.02)
ax[0].set_xlabel("eigenworm mode")
ax[0].set_ylabel("cumulative posture variance")
ax[0].set_title("a  Manifold = eigenworm space", loc="left", fontweight="bold")
ax[0].annotate(f"4 modes = {real_cv[-1]:.2f}", (4, real_cv[-1]),
               xytext=(2.2, 0.55), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="grey"))
ax[0].text(0.05, 0.10, "models align to real basis:\nmodWorm cos 0.77 · BAAIWorm cos 0.68",
           transform=ax[0].transAxes, fontsize=7.5, style="italic", color="#555")

# (b) complementary tiling: chemo vs motor param, in chemotaxis vs locomotion
c = chem["conditions"]
groups = ["chemotaxis\n(food close)", "locomotion\n(weak gradient)"]
chemo_vals = [c["close"]["chemo_perturb_behav_change"], c["far"]["chemo_perturb_behav_change"]]
motor_vals = [c["close"]["motor_perturb_behav_change"], c["far"]["motor_perturb_behav_change"]]
x = np.arange(2); w = 0.36
ax[1].bar(x - w/2, chemo_vals, w, color="#c44e52", label="chemosensory params")
ax[1].bar(x + w/2, motor_vals, w, color="#4c72b0", label="motor params")
ax[1].set_xticks(x); ax[1].set_xticklabels(groups)
ax[1].set_ylabel("behavioural sensitivity")
ax[1].set_title("b  Complementary tiling", loc="left", fontweight="bold")
ax[1].legend(fontsize=7.5, frameon=False, loc="upper right")
ax[1].set_ylim(0, max(chemo_vals + motor_vals) * 1.35)
ax[1].text(0.03, 0.97, f"chemosensory params:\n{chem['chemo_food_vs_nofood_ratio']:.1f}× context shift",
           transform=ax[1].transAxes, ha="left", va="top", fontsize=7.5, style="italic", color="#555")

# (c) modWorm saturation: union eff-dim vs #behaviours
nb  = [r["n_behaviours"] for r in sat["saturation_curve"]]
e99 = [r["eff_dim_99_mean"] for r in sat["saturation_curve"]]
ax[2].plot(nb, e99, "o-", color="#2a7", lw=2, ms=5)
ax[2].axhline(4, ls=":", color="grey", lw=1)
ax[2].set_xlabel("number of behaviours")
ax[2].set_ylabel("union effective dimension")
ax[2].set_title("c  Identifiability saturates (worm)", loc="left", fontweight="bold")
ax[2].text(0.95, 0.12, "residual 2–3 dirs\n= degenerate core",
           transform=ax[2].transAxes, ha="right", fontsize=7.5, style="italic", color="#555")

# (d) cross-species: union > single (richer repertoire -> larger union)
rod_curve = rod["saturation_curve"]
rnb = [r["n_behaviours"] for r in rod_curve]
re99 = [r["eff_dim_99_mean"] for r in rod_curve]
ax[3].plot(rnb, re99, "s-", color="#8856a7", lw=2, ms=5, label="rodent (38 act.)")
ax[3].plot(nb, e99, "o-", color="#2a7", lw=2, ms=4, label="worm (7 mech.)")
# larva: single mean -> union (2 points)
lar_single = np.mean([v["eff_dim_99"] for v in lar["per_behaviour"].values()])
lar_union = lar["union_eff_dim_90_99"][1]
ax[3].plot([1, len(lar["per_behaviour"])], [lar_single, lar_union], "^--",
           color="#e8a", lw=2, ms=6, label="larva (10 par.)")
ax[3].set_xlabel("number of behaviours")
ax[3].set_ylabel("union effective dimension")
ax[3].set_title("d  Union grows across phyla", loc="left", fontweight="bold")
ax[3].legend(fontsize=7.5, frameon=False, loc="upper left")

plt.tight_layout()
out = os.path.join(D, "fig_p2_tiling_crossspecies.png")
plt.savefig(out, dpi=200, bbox_inches="tight")
print("WROTE", out)
