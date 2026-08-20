"""Redraw the internal-neural-state panel at print resolution.

The stored version was 143 dpi at its printed width, well under the 300 dpi
floor. Same numbers, same message, drawn in the palette used by every other
figure: the dynamics generated under behaviour-recovered parameters sit in the
same low-dimensional, moderately correlated regime as the real recordings.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "..", "results", "NEW_EXPERIMENTS_RESULTS.json"),
                   encoding="utf-8"))["experiments"]["b2_behaviour_to_neural"]["comparison"]

BLUE, GREY, LGREY, INK = "#0072B2", "#808080", "#C9C9C9", "#1A1A1A"
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 600, "figure.dpi": 600,
})

arms = d["arms"]
keys = ["default", "pgob_real", "pgob_sim"]
lab = ["default", "recovered\n(real target)", "recovered\n(model target)"]
pdim = [arms[k]["pca_participation_dim"] for k in keys]
corr = [arms[k]["mean_abs_corr"] for k in keys]
kato = d["kato_real_mean_abs_corr"]

fig, ax = plt.subplots(1, 2, figsize=(89 * MM, 52 * MM))
x = np.arange(3)

ax[0].bar(x, pdim, color=[GREY, BLUE, BLUE], width=0.62)
ax[0].set_xticks(x); ax[0].set_xticklabels(lab, fontsize=5.5)
ax[0].set_ylabel("Participation dimension")
ax[0].set_ylim(0, 6)
ax[0].text(0.5, 5.55, "of 136 neurons", ha="center", fontsize=5.8, color=GREY)

ax[1].bar(x, corr, color=[GREY, BLUE, BLUE], width=0.62)
ax[1].axhline(kato, ls=(0, (3, 3)), color=INK, lw=0.7)
ax[1].text(2.45, kato + 0.012, "real recording", ha="right", fontsize=5.5, color=INK)
ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=5.5)
ax[1].set_ylabel("Mean |pairwise correlation|")
ax[1].set_ylim(0, 0.48)

for a in ax:
    a.tick_params(axis="x", length=0)
fig.tight_layout(w_pad=1.6)
p = os.path.join(HERE, "..", "paper", "figures", "fig_p2_b2_neural.png")
fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)

from PIL import Image
w, h = Image.open(p).size
print(f"fig_p2_b2_neural.png  {w}x{h}  = {w/(84/25.4):.0f} dpi at its printed width")
