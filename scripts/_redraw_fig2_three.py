"""Figure 2: the two connectome-scale measurements, and what sets the number.

Panel a is the distribution of per-connection sensitivity in the nematode model,
b is how its curvature divides between chemical synapses and gap junctions, and c
is the fly visual model measured over all three of its parameter groups together
with the readout-count dependence that explains why a bias-only probe looks
high-dimensional. Everything plotted is a stored measurement; no curve is fitted.
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
    "legend.fontsize": 5.8, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "savefig.dpi": 600, "figure.dpi": 600,
})


def panel(ax, letter, dx=-0.22, dy=1.10):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


b = L("E42_baaiworm_connectome.json")
fv = L("E4_flyvis_readout.json")

fig = plt.figure(figsize=(183 * MM, 56 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.62, 1.23], wspace=0.42)
ax = [fig.add_subplot(gs[0, i]) for i in range(3)]

# ---- a: per-connection sensitivity in the nematode connectome --------------
e90, e99 = b["eff_dim_90"], b["eff_dim_99"]
pr = b["participation_ratio"]
q = sorted((float(k), v) for k, v in b["elasticity_percentiles"].items())
ax[0].plot([t[0] for t in q], [t[1] for t in q], "o-", color=BLUE, lw=1.3, ms=3.6,
           markeredgecolor="white", markeredgewidth=0.4)
ax[0].set_yscale("log")
ax[0].set_xlabel("Percentile of connections")
ax[0].set_ylabel("Behavioural elasticity (log)")
ax[0].set_xlim(0, 100)
ax[0].text(0.04, 0.95, f"nematode connectome" + NL + f"{e90} of 3076 constrained",
           transform=ax[0].transAxes, fontsize=6, color=INK, va="top",
           linespacing=1.35)
panel(ax[0], "a")

# ---- b: which connection type carries the curvature ------------------------
cat = b["by_connection_category"]
names, shares, counts = [], [], []
for k, lab in (("syn", "chemical" + NL + "synapse"), ("gj", "gap" + NL + "junction")):
    if k in cat:
        names.append(lab)
        shares.append(cat[k]["share_of_curvature"])
        counts.append(cat[k]["n"])
y = np.arange(len(names))
ax[1].barh(y, shares, color=[VERM, GREEN][:len(names)], height=0.72)
ax[1].set_yticks(y)
ax[1].set_yticklabels([f"{n}{NL}({c})" for n, c in zip(names, counts)], fontsize=5.6)
ax[1].invert_yaxis()
ax[1].set_xlabel("Share of behavioural curvature")
ax[1].set_xlim(0, 1.0)
for yy, sh in zip(y, shares):
    ax[1].text(sh + 0.03, yy, f"{100*sh:.0f}%", va="center", fontsize=6, color=INK)
panel(ax[1], "b", dx=-0.40)

# ---- c: the fly visual model, and what the readout count does --------------
sw = fv["readout_sweep"]
ro = sorted(int(k) for k in sw)
ed = [sw[str(k)]["eff_dim_90"] for k in ro]
ax[2].plot(ro, ed, "o-", color=VERM, lw=1.3, ms=3.6,
           markeredgecolor="white", markeredgewidth=0.4,
           label="biases only")
ax[2].plot(ro, ro, ls=(0, (3, 3)), color=LGREY, lw=0.8, label="one dimension per readout")
joint = fv["all_groups_joint"]
ax[2].axhline(joint["eff_dim_90"], color=BLUE, lw=1.1,
              label=f"all three groups: {joint['eff_dim_90']} of "
                    f"{joint['n_params_probed']}")
ax[2].set_xscale("log", base=2)
ax[2].set_xticks(ro)
ax[2].set_xticklabels([str(r) for r in ro])
ax[2].set_xlabel("Output channels recorded")
ax[2].set_ylabel("Constrained dimensions")
ax[2].set_ylim(0, 72)
ax[2].legend(frameon=False, loc="upper left", handlelength=1.5,
             borderaxespad=0.2, labelspacing=0.35)
panel(ax[2], "c", dx=-0.24)

fig.tight_layout(w_pad=2.4)
p = os.path.join(FIG, "Fig_connectome_curvature.png")
fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)
from PIL import Image
w_, h_ = Image.open(p).size
print(f"  Fig_connectome_curvature.png  {w_}x{h_}")
print(f"  flyvis readout sweep {list(zip(ro, ed))}")
print(f"  flyvis all groups {joint['eff_dim_90']}/{joint['eff_dim_99']} of "
      f"{joint['n_params_probed']}")
