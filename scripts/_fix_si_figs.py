"""Repair the three supplementary figures whose annotation collided with the data.

Three separate faults, same cause: text was placed at a fixed axes fraction without
checking what the data does there.

  fig_p2_b3_localise      every label anchored at the same offset, so the four
                          points sharing a y value wrote over each other and the
                          near-silent cluster wrote over its own caption
  fig_p2_crosssystem_real the group label sat under the longest bar, which is drawn
                          on top of it, leaving only the last glyph visible
  fig_p2_b2_neural        no panel letter, and two-line tick labels wide enough to
                          touch their neighbours

Numbers, colours and message are unchanged. Panel letters are drawn to match the
captions, which pair these images two to a float.
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

BLUE, GREY, LGREY, INK = "#0072B2", "#808080", "#C9C9C9", "#1A1A1A"
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 5.6, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 600, "figure.dpi": 600,
})


def letter(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(p).size
    print(f"  {name:30s} {w}x{h}  = {w/300*25.4:.0f} mm at 300 dpi")


# ---------------------------------------------------------- b3: mechanism localisation
d = L("baai_b3_localise.json")
rank = d["rank_by_signature_norm"]          # [key, name, signature_norm, n_confusable]
names = [r[1] for r in rank]
sig = np.array([float(r[2]) for r in rank])
conf = np.array([float(r[3]) for r in rank])
WIRING = ("gap", "synap", "motor", "leak", "passive", "nca")
cls = [BLUE if any(w in n.lower() for w in WIRING) else GREY for n in names]

fig, ax = plt.subplots(figsize=(89 * MM, 64 * MM))
ax.axvline(0.2, ls=(0, (3, 3)), color=LGREY, lw=0.6, zorder=1)
ax.scatter(sig, conf, s=16, c=cls, edgecolor="white", linewidth=0.3, zorder=3)

# label only what is above the detectability threshold, and stagger the labels
# within each y level so that points sharing a row never write over each other
for y in sorted(set(conf)):
    idx = [i for i in np.argsort(sig)[::-1] if conf[i] == y and sig[i] > 0.2]
    for j, i in enumerate(idx):
        up = (j % 2 == 0)
        ax.annotate(names[i], (sig[i], conf[i]), textcoords="offset points",
                    xytext=(-3, 5.5 if up else -9.5), ha="right",
                    va="bottom" if up else "top", fontsize=5, color=INK, zorder=4)

ax.set_xlabel("Behavioural signature norm (detectability)")
ax.set_ylabel("Number of confusable mechanisms")
ax.set_xlim(0, 1.42)
ax.set_ylim(-0.75, 4.85)
ax.set_yticks([0, 1, 2, 3, 4])
ax.text(0.02, 3.15, "left of the line:\nbehaviourally near-silent\n(9 channels, not labelled)",
        ha="left", va="center", fontsize=5.4, color=GREY, linespacing=1.35)
h = [plt.Line2D([], [], ls="", marker="o", ms=3.2, color=BLUE,
                markeredgecolor="white", markeredgewidth=0.3),
     plt.Line2D([], [], ls="", marker="o", ms=3.2, color=GREY,
                markeredgecolor="white", markeredgewidth=0.3)]
ax.legend(h, ["wiring / passive", "single-cell channels"], loc="center right",
          frameon=False, handletextpad=0.4, borderaxespad=0.3, labelspacing=0.45)
letter(ax, "b", x=-0.14)
fig.tight_layout()
save(fig, "fig_p2_b3_localise.png")


# --------------------------------------------------- crosssystem: pyloric loadings
d = L("STG_b1analog.json")
stiff = [(n, float(w)) for n, w in d["stiff_top"]][:6]
sloppy = [(n, float(w)) for n, w in d["sloppy_top"]][:6]
# Synapses.PY-LP carries loading in both eigenvectors; say which one each row is
sl_names = {n for n, _ in stiff}
sloppy = [(f"{n} (sloppy ev.)" if n in sl_names else n, w) for n, w in sloppy]
sel = stiff + sloppy

fig, ax = plt.subplots(figsize=(89 * MM, 60 * MM))
ax.barh(range(len(sel)), [t[1] for t in sel],
        color=[BLUE] * len(stiff) + [GREY] * len(sloppy), height=0.72, zorder=2)
ax.set_yticks(range(len(sel))); ax.set_yticklabels([t[0] for t in sel], fontsize=5.5)
ax.invert_yaxis()
ax.set_xlabel("Eigenvector loading")
ax.set_xlim(0, 1.16)                       # headroom so no annotation sits on a bar
ax.axhline(len(stiff) - 0.5, color=LGREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
# anchored to rows whose bars are short, so nothing is painted over
ax.text(1.14, 3.2, "stiff: leak and\ninhibitory synapses", ha="right", va="center",
        fontsize=5.8, color=BLUE, linespacing=1.25, zorder=4)
ax.text(1.14, 9.2, "sloppy: fast voltage-\ngated sodium", ha="right", va="center",
        fontsize=5.8, color=GREY, linespacing=1.25, zorder=4)
ax.text(1.14, 11.4, f"effective dimension {d['eff_dim_mean']:.2f} of 31",
        ha="right", va="center", fontsize=6, color=INK, zorder=4)
letter(ax, "b", x=-0.30)
fig.tight_layout()
save(fig, "fig_p2_crosssystem_real.png")


# ------------------------------------------------------- b2: internal neural state
d = json.load(open(os.path.join(RES, "NEW_EXPERIMENTS_RESULTS.json"),
                   encoding="utf-8"))["experiments"]["b2_behaviour_to_neural"]["comparison"]
arms = d["arms"]
keys = ["default", "pgob_real", "pgob_sim"]
lab = ["default", "recovered\n(real)", "recovered\n(model)"]
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
ax[1].set_xlim(-0.62, 3.30)          # room to label the line clear of the bars
ax[1].text(3.28, kato + 0.008, "real recording", ha="right", fontsize=5.5, color=INK)
ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=5.5)
ax[1].set_ylabel("Mean |pairwise correlation|")
ax[1].set_ylim(0, 0.48)

for a in ax:
    a.tick_params(axis="x", length=0)
letter(ax[0], "a", x=-0.26)
fig.tight_layout(w_pad=2.2)
save(fig, "fig_p2_b2_neural.png")


# the companion panel in the same float carries the matching letter
d = L("flyvis_b1.json")
st = d.get("per_celltype_stiffness", {})
items = sorted(((str(kk).replace("b'", "").replace("'", ""), abs(float(vv)))
                for kk, vv in st.items()), key=lambda t: t[1], reverse=True)
top, bot = items[:10], items[-10:]
sel = top + bot
fig, ax = plt.subplots(figsize=(89 * MM, 72 * MM))
ax.barh(range(len(sel)), [t[1] for t in sel],
        color=[BLUE] * len(top) + [GREY] * len(bot), height=0.72, zorder=2)
ax.set_yticks(range(len(sel))); ax.set_yticklabels([t[0] for t in sel], fontsize=5.5)
ax.invert_yaxis(); ax.set_xlabel("Behavioural elasticity")
xm = max(t[1] for t in sel)
ax.set_xlim(0, xm * 1.42)
ax.axhline(len(top) - 0.5, color=LGREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
ax.text(xm * 1.40, 12.6, f"effective dimension\n{d['eff_dim_90']} of {d['n_cell_types']} cell types",
        ha="right", va="center", fontsize=6, color=INK, linespacing=1.3, zorder=4)
ax.text(xm * 1.40, 1.0, "10 stiffest", ha="right", va="center", fontsize=5.8,
        color=BLUE, zorder=4)
ax.text(xm * 1.40, 15.0, "10 sloppiest", ha="right", va="center", fontsize=5.8,
        color=GREY, zorder=4)
letter(ax, "a", x=-0.24)
fig.tight_layout()
save(fig, "fig_p2_flyvis_b1.png")
