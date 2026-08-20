"""Redraw the supplementary flyvis panel around the corrected measurement.

The deposited version showed the per-cell-type stiffness of the bias group alone,
which is the probe that made the model look high-dimensional. This version shows
what each of the three free parameter groups contributes and where the joint
measurement lands, and keeps the per-cell-type ranking as the second panel since
the biological reading of which cell types matter is drawn from it.
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
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "savefig.dpi": 600, "figure.dpi": 600,
})

fv = L("E4_flyvis_readout.json")
g = fv["group_sweep"]
joint = fv["all_groups_joint"]

fig, ax = plt.subplots(figsize=(89 * MM, 56 * MM))

LABEL = {"nodes_bias": "per-cell-type bias",
         "nodes_time_const": "membrane time constant",
         "edges_syn_strength": "synaptic strength"}
keys = [k for k in ("nodes_bias", "nodes_time_const", "edges_syn_strength") if k in g]
lab = [LABEL[k] for k in keys] + ["all three groups"]
val = [g[k]["eff_dim_90"] for k in keys] + [joint["eff_dim_90"]]
npar = [g[k]["n_params_probed"] for k in keys] + [joint["n_params_probed"]]
col = [GREY] * len(keys) + [BLUE]

y = np.arange(len(lab))
ax.barh(y, val, color=col, height=0.55)
ax.set_yticks(y)
ax.set_yticklabels([f"{a}{NL}({b} parameters)" for a, b in zip(lab, npar)],
                   fontsize=5.6)
ax.invert_yaxis()
ax.set_xlabel("Constrained dimensions at 90% of the curvature")
for yy, v, npp in zip(y, val, npar):
    ax.text(v + 0.8, yy, f"{v} of {npp}", va="center", fontsize=5.8, color=INK)
ax.set_xlim(0, max(val) * 1.35)
ax.text(0.97, 0.06,
        "the bias group alone is what makes" + NL
        + "the model look high dimensional",
        transform=ax.transAxes, ha="right", fontsize=5.8, color=GREY,
        linespacing=1.3)
fig.tight_layout()
p = os.path.join(FIG, "fig_p2_flyvis_b1.png")
fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)
from PIL import Image
w_, h_ = Image.open(p).size
print(f"  fig_p2_flyvis_b1.png  {w_}x{h_}")
for a, v, npp in zip(lab, val, npar):
    print(f"    {a:26s} {v} of {npp}")
