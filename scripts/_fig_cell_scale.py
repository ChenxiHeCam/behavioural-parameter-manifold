"""SI figure for S7.4: network size does not dilute the constrained fraction.

Panel a  Constrained fraction against parameter count as identical
         Hodgkin-Huxley cells are added to a fixed sparse network: essentially
         flat from 6 to 192 parameters, at both observable budgets.

Panel b  The effective dimension itself, rising linearly at close to one
         constrained direction per cell (dashed reference), so the flat
         fraction is not a ceiling artefact.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "paper", "figures")

BLUE, GREY, INK, LGREY = "#0072B2", "#808080", "#1A1A1A", "#C9C9C9"
MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 5.8, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 600, "figure.dpi": 600,
})

rows = json.load(open(os.path.join(W, "cell_scale_result.json")))["rows"]
series = {}
for r in rows:
    series.setdefault(r["obs_per_param_target"], []).append(
        (r["n_params"], r["constrained_fraction_90"], r["eff_dim_90"]))
for k in series:
    series[k].sort()

fig, ax = plt.subplots(1, 2, figsize=(150 * MM, 52 * MM))

a = ax[0]
for (ratio, col, lab) in ((10, GREY, "10 observables per parameter"),
                          (30, BLUE, "30 observables per parameter")):
    d = np.array(series[ratio])
    a.plot(d[:, 0], d[:, 1], "o-", color=col, lw=1.1, ms=3.2,
           markeredgecolor="white", markeredgewidth=0.4, label=lab)
a.set_xscale("log")
a.set_xlabel("Number of parameters (3 per cell)")
a.set_ylabel("Constrained fraction (90%)")
a.set_ylim(0, 0.75)
a.set_xticks([3, 12, 48, 192])
a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
a.legend(frameon=False, loc="upper right", handletextpad=0.5)
a.text(-0.16, 1.04, "a", transform=a.transAxes, fontsize=8, fontweight="bold")

b = ax[1]
d = np.array(series[30])
b.plot(d[:, 0], d[:, 2], "o-", color=BLUE, lw=1.1, ms=3.2,
       markeredgecolor="white", markeredgewidth=0.4)
b.plot(d[:, 0], d[:, 0] / 3.0, ls=(0, (3, 3)), color=LGREY, lw=0.8)
b.set_xscale("log")
b.set_yscale("log")
b.set_xlabel("Number of parameters (3 per cell)")
b.set_ylabel("Effective dimension (90%)")
b.set_xticks([3, 12, 48, 192])
b.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.set_yticks([1, 3, 10, 30])
b.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
b.text(70, 5.0, "one direction\nper cell", fontsize=5.8, color=GREY,
       linespacing=1.25, ha="left")
b.text(-0.18, 1.04, "b", transform=b.transAxes, fontsize=8, fontweight="bold")

fig.tight_layout(w_pad=2.6)
p = os.path.join(FIG, "fig_p2_cell_scale.png")
fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)
from PIL import Image
w, h = Image.open(p).size
print(f"fig_p2_cell_scale.png  {w}x{h}  = {w/300*25.4:.0f} mm at 300 dpi")
