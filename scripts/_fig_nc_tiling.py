"""Nature Communications-style hero figure (Fig. 3): eigenworm identity, complementary tiling,
union growth, cross-phyla. Print conventions: 180 mm double-column width, sans-serif ~7 pt,
bold lowercase panel labels outside the axes, no in-panel titles (they live in the caption),
CVD-safe Okabe-Ito palette (validated), thin recessive axes, no gridlines, direct labels."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "paper", "figures", "fig_p2_tiling_crossspecies.png")
L = lambda n: json.load(open(os.path.join(D, n)))

ew, chem = L("EW_eigenworm.json"), L("BAAI_chemo.json")
sat, rod, lar = L("SAT_saturation.json"), L("RODENT_manifold.json"), L("LARVA_manifold.json")

# --- validated CVD-safe palette (Okabe-Ito subset; all six checks PASS) ---
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#c9c9c9"

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "figure.dpi": 600, "savefig.dpi": 600,
})

fig, ax = plt.subplots(1, 4, figsize=(180 * MM, 45 * MM))
plt.subplots_adjust(left=0.055, right=0.995, top=0.86, bottom=0.24, wspace=0.42)

# ---- a: eigenworm cumulative variance ----
cv = ew["real_varexp_top4"]
ax[0].plot(range(1, 5), cv, "o-", color=BLUE, lw=1.4, ms=4, clip_on=False, zorder=3)
ax[0].axhline(0.9, ls=(0, (3, 3)), color=GRID, lw=0.6, zorder=1)
ax[0].set_xticks(range(1, 5)); ax[0].set_ylim(0, 1.05)
ax[0].set_xlabel("Eigenworm mode"); ax[0].set_ylabel("Cumulative variance")
ax[0].annotate(f"{cv[-1]:.2f}", xy=(4, cv[-1]), xytext=(-2, -9),
               textcoords="offset points", ha="right", fontsize=6.5, color=MUTED)

# ---- b: complementary tiling ----
c = chem["conditions"]
x = np.arange(2); w = 0.32
ax[1].bar(x - w/2 - 0.012, [c["close"]["chemo_perturb_behav_change"], c["far"]["chemo_perturb_behav_change"]],
          w, color=VERM, label="Chemosensory", zorder=3)
ax[1].bar(x + w/2 + 0.012, [c["close"]["motor_perturb_behav_change"], c["far"]["motor_perturb_behav_change"]],
          w, color=BLUE, label="Motor", zorder=3)
ax[1].set_xticks(x); ax[1].set_xticklabels(["Chemotaxis", "Locomotion"])
ax[1].set_ylabel("Behavioural sensitivity")
ax[1].set_ylim(0, 1.08)
ax[1].legend(frameon=False, loc="upper right", handlelength=1.0, borderpad=0.2, labelspacing=0.25)

# ---- c: union saturation (worm) ----
nb = [r["n_behaviours"] for r in sat["saturation_curve"]]
e99 = [r["eff_dim_99_mean"] for r in sat["saturation_curve"]]
ax[2].plot(nb, e99, "o-", color=GREEN, lw=1.4, ms=3.2, clip_on=False, zorder=3)
ax[2].axhline(4, ls=(0, (3, 3)), color=GRID, lw=0.6, zorder=1)
ax[2].set_xlabel("Number of behaviours"); ax[2].set_ylabel("Union effective dimension")
ax[2].set_xticks([1, 4, 8, 12]); ax[2].set_ylim(1, 4.4)

# ---- d: union growth across phyla (direct labels, no legend box) ----
rc = rod["saturation_curve"]
rnb = [r["n_behaviours"] for r in rc]; re99 = [r["eff_dim_99_mean"] for r in rc]
lar_single = float(np.mean([v["eff_dim_99"] for v in lar["per_behaviour"].values()]))
lar_union = lar["union_eff_dim_90_99"][1]
ax[3].plot(rnb, re99, "s-", color=VERM, lw=1.4, ms=3.2, clip_on=False, zorder=3)
ax[3].plot([1, len(lar["per_behaviour"])], [lar_single, lar_union], "^-", color=BLUE,
           lw=1.4, ms=3.6, clip_on=False, zorder=3)
ax[3].plot(nb, e99, "o-", color=GREEN, lw=1.4, ms=3.0, clip_on=False, zorder=3)
ax[3].annotate("Rodent", xy=(6, 12.0), xytext=(5, 6), textcoords="offset points",
               color=VERM, fontsize=6.5, va="center", ha="left")
ax[3].annotate("Larva", xy=(4, lar_union), xytext=(5, -7), textcoords="offset points",
               color=BLUE, fontsize=6.5, va="center", ha="left")
ax[3].annotate("Worm", xy=(12, 4.0), xytext=(5, 0), textcoords="offset points",
               color=GREEN, fontsize=6.5, va="center", ha="left")
ax[3].set_xlabel("Number of behaviours"); ax[3].set_ylabel("Union effective dimension")
ax[3].set_xlim(0.5, 15.5); ax[3].set_ylim(0, 13.5); ax[3].set_xticks([1, 4, 8, 12])

# ---- panel labels: bold lowercase, outside axes (NC convention) ----
for a, lab in zip(ax, "abcd"):
    a.text(-0.26, 1.10, lab, transform=a.transAxes, fontsize=8,
           fontweight="bold", va="top", ha="left")

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("WROTE", OUT)
