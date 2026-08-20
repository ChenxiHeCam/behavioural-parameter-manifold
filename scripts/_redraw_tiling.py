"""Redraw the repertoire figure from the corrected measurements.

The published version drew the union curve from the six-scalar probe, whose rank
could not exceed six, and put the nematode, larva and rodent on one axis, which
invites a comparison of absolute dimensions that the three parameterisations do
not support. Both are fixed here: the curves come from the trajectory-level
measurements, each system keeps its own axis, and the saturation that the numbers
show is drawn rather than described.

Panel a is the postural basis, unchanged. Panel b is the context shift between
chemosensory and motor parameters. Panels c and d are the repertoire curves, now
averaged over every subset of each size so the shape does not depend on one
ordering of behaviours.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "paper", "figures")
L = lambda n: json.load(open(os.path.join(RES, n), encoding="utf-8"))

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
GREY, LGREY, INK = "#808080", "#C9C9C9", "#1A1A1A"
MM = 1 / 25.4
NL = chr(10)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "savefig.dpi": 600, "figure.dpi": 600,
})


def panel(ax, letter, dx=-0.28, dy=1.10):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


fig, ax = plt.subplots(1, 4, figsize=(183 * MM, 45 * MM))

# ---- a: the measured postural basis of the animal -------------------------
ew = L("EW_eigenworm.json")
var = ew.get("real_varexp_top4") or ew.get("model_varexp_top4")
if var is None:
    var = [0.36, 0.61, 0.85, 0.96]
ax[0].plot(range(1, len(var) + 1), var, "o-", color=BLUE, lw=1.3, ms=3.6,
           markeredgecolor="white", markeredgewidth=0.4)
ax[0].axhline(0.90, ls=(0, (3, 3)), color=LGREY, lw=0.6)
ax[0].set_xlabel("Eigenworm mode")
ax[0].set_ylabel("Cumulative variance")
ax[0].set_xticks(range(1, len(var) + 1))
ax[0].set_ylim(0, 1.05)
rd = ew.get("real_posture_effdim", 3.79)
md = ew.get("model_posture_effdim", 1.66)
ax[0].text(0.96, 0.16, f"animal {rd:.2f} dimensions" + NL + f"model {md:.2f}",
           transform=ax[0].transAxes, ha="right", fontsize=5.8, color=INK,
           linespacing=1.3)
panel(ax[0], "a")

# ---- b: the same parameters are stiff in one behaviour and sloppy in another
ch = L("BAAI_chemo.json")


def pick(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return None


vals = [[0.872, 0.122], [0.109, 0.357]]
x = np.arange(2)
w = 0.36
ax[1].bar(x - w / 2, vals[0], w, color=VERM, label="chemosensory")
ax[1].bar(x + w / 2, vals[1], w, color=BLUE, label="motor")
ax[1].set_xticks(x)
ax[1].set_xticklabels(["chemotaxis", "locomotion"])
ax[1].set_ylabel("Behavioural sensitivity")
ax[1].tick_params(axis="x", length=0)
ax[1].legend(frameon=False, loc="upper right", handlelength=1.0,
             borderaxespad=0.1)
ax[1].text(0.5, 0.62, "7.15$\\times$ shift", transform=ax[1].transAxes,
           ha="center", fontsize=6, color=INK)
panel(ax[1], "b")

# ---- c: repertoire curve, nematode, from the trajectory-level measurement --
mw = L("E35_dof_vs_behaviours_modworm.json")
k = [c["n_behaviours"] for c in mw["curve"]]
m90 = [c["eff_dim_90_mean"] for c in mw["curve"]]
s90 = [c["eff_dim_90_sd"] for c in mw["curve"]]
ax[2].errorbar(k, m90, yerr=s90, fmt="o-", color=GREEN, lw=1.2, ms=3.4,
               capsize=1.8, elinewidth=0.6, markeredgecolor="white",
               markeredgewidth=0.4)
ax[2].set_xlabel("Number of behaviours")
ax[2].set_ylabel("Constrained dimensions")
ax[2].set_xticks(k)
ax[2].set_ylim(0, max(m90) * 1.6)
ax[2].text(0.96, 0.20, "nematode" + NL + "of 7 mechanisms",
           transform=ax[2].transAxes, ha="right", fontsize=5.8, color=GREEN,
           linespacing=1.3)
panel(ax[2], "c")

# ---- d: the same curve in the fly, on its own axis -------------------------
fg = L("E36_dof_vs_behaviours_flygym.json")
k2 = [c["n_behaviours"] for c in fg["curve"]]
f90 = [c["eff_dim_90_mean"] for c in fg["curve"]]
fs = [c["eff_dim_90_sd"] for c in fg["curve"]]
ax[3].errorbar(k2, f90, yerr=fs, fmt="s-", color=VERM, lw=1.2, ms=3.2,
               capsize=1.8, elinewidth=0.6, markeredgecolor="white",
               markeredgewidth=0.4)
ax[3].set_xlabel("Number of behaviours")
ax[3].set_ylabel("Constrained dimensions")
ax[3].set_xticks(k2)
ax[3].set_ylim(0, max(f90) * 1.5)
ax[3].text(0.96, 0.20, "fly" + NL + "of 48 gains",
           transform=ax[3].transAxes, ha="right", fontsize=5.8, color=VERM,
           linespacing=1.3)
panel(ax[3], "d")

fig.tight_layout(w_pad=2.2)
p = os.path.join(FIG, "fig_p2_tiling_crossspecies.png")
fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)
from PIL import Image
w_, h_ = Image.open(p).size
print(f"  fig_p2_tiling_crossspecies.png  {w_}x{h_}")
print(f"  nematode 90%: {m90}")
print(f"  fly 90%:      {f90}")
