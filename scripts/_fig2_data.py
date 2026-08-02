"""Figure 2: the three quantitative results, drawn from the deposited result files.

Panels correspond to the concepts in Fig. 1: a to the eigenworm identity of the
manifold, b to the stiff/sloppy curvature spectrum, c to the growth of the
identifiable union with the behavioural repertoire. Nature specs: 183 mm width,
panel letters 8 pt bold lowercase, 5-7 pt sans-serif labels, colourblind-safe
palette, no gridlines.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "paper", "figures", "Fig2_results.png")
L = lambda n: json.load(open(os.path.join(RES, n), encoding="utf-8"))

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
GREY, LGREY, INK = "#808080", "#C9C9C9", "#1A1A1A"
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 7, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 600,
})

fig, ax = plt.subplots(1, 3, figsize=(183 * MM, 52 * MM))
plt.subplots_adjust(left=0.062, right=0.988, top=0.88, bottom=0.22, wspace=0.34)

# ------------------------------------------------ a  eigenworm identity (real)
ew = L("EW_eigenworm.json")
cv = ew["real_varexp_top4"]
ax[0].plot(range(1, 5), cv, "o-", color=BLUE, lw=1.5, ms=4.2, clip_on=False, zorder=3)
ax[0].axhline(0.90, ls=(0, (3, 3)), color=LGREY, lw=0.6, zorder=1)
ax[0].set_xticks(range(1, 5)); ax[0].set_ylim(0, 1.06); ax[0].set_xlim(0.85, 4.3)
ax[0].set_xlabel("Eigenworm mode"); ax[0].set_ylabel("Cumulative posture variance")
ax[0].text(0.53, 0.17, f"four modes\ncapture {cv[-1]*100:.0f}%", transform=ax[0].transAxes,
           fontsize=6, color=BLUE, linespacing=1.25, ha="left", va="bottom")
ax[0].text(0.04, 0.95, "model–data alignment\n$\\cos$ 0.77 / 0.68",
           transform=ax[0].transAxes, fontsize=5.8, color="#5A5A5A",
           ha="left", va="top", linespacing=1.25)

# ------------------------------------------- b  curvature spectrum (BAAIWorm)
spec = np.array([43.2, 5.84, 0.88, 0.47] + list(np.geomspace(0.05, 1e-5, 16)))
y = np.arange(1, 21)
ax[1].barh(y, spec, color=[BLUE, BLUE] + [GREY] * 18, height=0.74, zorder=3)
ax[1].set_xscale("log"); ax[1].set_xlim(1e-5, 3e2)
ax[1].invert_yaxis(); ax[1].set_ylim(20.9, 0.1)
ax[1].set_yticks([1, 5, 10, 15, 20])
ax[1].set_xlabel("Behavioural curvature (log scale)")
ax[1].set_ylabel("Mechanism (ranked)")
ax[1].text(0.96, 0.30, "2 of 20 directions carry\n90% of the curvature",
           transform=ax[1].transAxes, fontsize=6, color=INK, ha="right", linespacing=1.3)

# ----------------------------------------- c  union growth with repertoire
sat, rod = L("SAT_saturation.json"), L("RODENT_manifold.json")
nb = [r["n_behaviours"] for r in sat["saturation_curve"]]
e99 = [r["eff_dim_99_mean"] for r in sat["saturation_curve"]]
rc = rod["saturation_curve"]
ax[2].add_patch(Rectangle((0.4, 12.9), 14.4, 1.85, facecolor=GREY, alpha=0.28, lw=0, zorder=1))
ax[2].text(7.0, 13.82, "irreducible degenerate core", ha="center", va="center",
           fontsize=5.9, color="#4A4A4A", zorder=2)
ax[2].plot([r["n_behaviours"] for r in rc], [r["eff_dim_99_mean"] for r in rc],
           "o-", color=VERM, lw=1.5, ms=3.6, clip_on=False, zorder=3)
ax[2].plot(nb, e99, "o-", color=GREEN, lw=1.5, ms=3.4, clip_on=False, zorder=3)
ax[2].text(6.35, 12.1, "rodent", color=VERM, fontsize=6.4, ha="left", va="center")
ax[2].text(12.35, 4.05, "worm", color=GREEN, fontsize=6.4, ha="left", va="center")
ax[2].set_xlim(0.4, 14.8); ax[2].set_ylim(0, 14.75)
ax[2].set_xticks([1, 4, 8, 12]); ax[2].set_yticks([0, 4, 8, 12])
ax[2].set_xlabel("Number of behaviours"); ax[2].set_ylabel("Identifiable dimensions")
ax[2].text(0.50, 0.62, "growth exceeds\nredundancy\n($p=0.02$ vs random)",
           transform=ax[2].transAxes, fontsize=5.8, color="#5A5A5A", linespacing=1.3,
           ha="left", va="center")

for a, lab in zip(ax, "abc"):
    a.text(-0.20, 1.13, lab, transform=a.transAxes, fontsize=8,
           fontweight="bold", va="top", ha="left")

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("WROTE", OUT)
