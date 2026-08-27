"""Compose the three figures that were two files placed side by side.

Setting two separate images next to each other with \\hfill is not a panelled
figure: the baselines do not align, the panel heights are set by whatever each
script happened to choose, the two axes carry independent font metrics, and a
journal that asks for one file per figure receives two. These three are rebuilt
as single figures with a shared grid, one panel letter each, and a common style.

  Fig 4 (main)  BAAIWorm named conductances | FlyGym eigenspectrum
  SI  b2/b3     internal neural state       | knockdown detectability
  SI  flyvis    fly optic-lobe elasticity   | pyloric eigenvector loadings
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


def letter(ax, s, x, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left", color=INK)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(p).size
    print(f"  {name:34s} {w}x{h}  = {w/300*25.4:.0f} mm at 300 dpi")


# ============================ Fig 4: named conductances | FlyGym spectrum =====
fig, ax = plt.subplots(1, 2, figsize=(183 * MM, 62 * MM),
                       gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.42})

c = L("E45b_baai_named_channels.json")   # mechanism-lookup fix; supersedes E45
rank = c["ranking_stiff_to_sloppy"]
nm = [r[0] for r in rank]
val = np.array([r[1] for r in rank], dtype=float)
WIRING = ("leak", "nca", "irk")
col = [BLUE if any(w in n.lower() for w in WIRING) else GREY for n in nm]
a = ax[0]
y = np.arange(len(val))
a.hlines(y, val.min() * 0.5, val, color=col, lw=1.0, alpha=0.55)
a.scatter(val, y, s=12, c=col, zorder=3, edgecolor="white", linewidth=0.3)
a.set_yticks(y)
a.set_yticklabels(nm, fontsize=5.4)
a.invert_yaxis()
a.set_xscale("log")
a.set_xlabel("Behavioural elasticity (log scale)")
a.text(0.97, 0.08, f"effective dimension {c['eff_dim_90']} of {c['n_mechanisms']}",
       transform=a.transAxes, ha="right", fontsize=6, color=INK)
letter(a, "d", -0.34)

sc = L("AB4_flygym_delta.json")
rows = sorted(sc["by_delta"], key=lambda r: r["delta"])
dl = [r["delta"] for r in rows]
d90 = [r["rich"]["eff_dim_90"] for r in rows]
b = ax[1]
b.axvspan(0.20, 0.55, color="#F0F0F0", zorder=0)
b.plot(dl, d90, "o-", color=BLUE, lw=1.1, ms=3.4,
       markeredgecolor="white", markeredgewidth=0.4, zorder=3, clip_on=False)
b.axhline(11, ls=(0, (3, 3)), color=LGREY, lw=0.7, zorder=1)
b.set_xscale("log")
b.set_xticks(dl)
b.get_xaxis().set_major_formatter(__import__("matplotlib").ticker.ScalarFormatter())
b.set_xlabel("Perturbation step")
b.set_ylabel("Effective dimension (90%)")
b.set_ylim(9.5, 17)
b.annotate("below the proportional range", xy=(0.05, 16), xytext=(0.075, 16.4),
           fontsize=5.8, color=GREY, ha="left", va="center",
           arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY))
b.text(0.36, 11.55, "11 across the proportional range", fontsize=6, color=INK,
       ha="center", linespacing=1.25)
b.text(0.02, 0.04, f"{sc['n_rich_observables']} observables against "
                   f"{sc['n_params']} parameters",
       transform=b.transAxes, ha="left", va="bottom", fontsize=6, color=INK)
letter(b, "e", -0.16)
save(fig, "Fig_channels_flygym.png")


# =================== SI: internal neural state | knockdown detectability =====
fig = plt.figure(figsize=(183 * MM, 58 * MM))
gs = fig.add_gridspec(1, 3, width_ratios=[0.48, 0.48, 1.70], wspace=0.58)
a0, a1, a2 = (fig.add_subplot(gs[0, i]) for i in range(3))

d = json.load(open(os.path.join(RES, "NEW_EXPERIMENTS_RESULTS.json"),
                   encoding="utf-8"))["experiments"]["b2_behaviour_to_neural"]["comparison"]
arms = d["arms"]
keys = ["default", "pgob_real", "pgob_sim"]
lab = ["default", "real", "model"]     # recovered from behaviour; see caption
x = np.arange(3)
a0.bar(x, [arms[k]["pca_participation_dim"] for k in keys],
       color=[GREY, BLUE, BLUE], width=0.62)
a0.set_xticks(x); a0.set_xticklabels(lab, fontsize=5.6)
a0.set_ylabel("Participation dimension")
a0.set_ylim(0, 6)
a0.text(1.0, 5.6, "of 136 neurons", ha="center", fontsize=5.6, color=GREY)
a0.tick_params(axis="x", length=0)
letter(a0, "a", -0.42)

kato = d["kato_real_mean_abs_corr"]
a1.bar(x, [arms[k]["mean_abs_corr"] for k in keys],
       color=[GREY, BLUE, BLUE], width=0.62)
a1.axhline(kato, ls=(0, (3, 3)), color=INK, lw=0.7)
a1.text(-0.45, kato + 0.012, "real recording", ha="left", fontsize=5.2, color=INK)
a1.set_xticks(x); a1.set_xticklabels(lab, fontsize=5.6)
a1.set_ylabel("Mean |pairwise correlation|")
a1.set_ylim(0, 0.48)
a1.tick_params(axis="x", length=0)

e = L("baai_b3_localise.json")
rk = e["rank_by_signature_norm"]
names = [x_[1] for x_ in rk]
sig = np.array([float(x_[2]) for x_ in rk])
conf = np.array([float(x_[3]) for x_ in rk])
W2 = ("gap", "synap", "motor", "leak", "passive", "nca")
cls = [BLUE if any(w in n.lower() for w in W2) else GREY for n in names]
a2.axvline(0.2, ls=(0, (3, 3)), color=LGREY, lw=0.6, zorder=1)
a2.scatter(sig, conf, s=16, c=cls, edgecolor="white", linewidth=0.3, zorder=3)
# Only the three mechanisms the text names are labelled. Eighteen names in this
# space cannot be placed without collisions, and the figure's content is the
# pattern, not the roll-call: the caption carries the individual values.
KEY = {"synaptic-weight": (0, 8, "center", "bottom"),
       "passive-leak": (-6, 0, "right", "center"),
       "motor-output": (0, -9, "center", "top")}
for i, nm_ in enumerate(names):
    if nm_ in KEY:
        dx, dy, ha, va = KEY[nm_]
        a2.annotate(nm_, (sig[i], conf[i]), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va, fontsize=5.4, color=INK,
                    zorder=4)
a2.set_xlabel("Behavioural signature norm (detectability)")
a2.set_ylabel("Number of confusable mechanisms")
a2.set_xlim(0, 1.42); a2.set_ylim(-0.75, 4.85)
a2.set_yticks([0, 1, 2, 3, 4])
a2.text(0.02, 2.95, "left of the line:\nbehaviourally near-silent\n"
                    "(9 channels, not labelled)",
        ha="left", va="center", fontsize=5.2, color=GREY, linespacing=1.35)
h = [plt.Line2D([], [], ls="", marker="o", ms=3.2, color=c_,
                markeredgecolor="white", markeredgewidth=0.3) for c_ in (BLUE, GREY)]
a2.legend(h, ["wiring / passive", "single-cell channels"], loc="center",
          bbox_to_anchor=(0.78, 0.60), frameon=False, handletextpad=0.4,
          borderaxespad=0.0, labelspacing=0.45)
letter(a2, "b", -0.15)
save(fig, "Fig_si_neural_localise.png")


# ==================== SI: flyvis elasticity | pyloric eigenvector loadings ====
fig, ax = plt.subplots(1, 2, figsize=(183 * MM, 76 * MM),
                       gridspec_kw={"wspace": 0.62})

d = L("flyvis_b1.json")
st = d.get("per_celltype_stiffness", {})
items = sorted(((str(k).replace("b'", "").replace("'", ""), abs(float(v)))
                for k, v in st.items()), key=lambda t: t[1], reverse=True)
top, bot = items[:10], items[-10:]
sel = top + bot
a = ax[0]
a.barh(range(len(sel)), [t[1] for t in sel],
       color=[BLUE] * len(top) + [GREY] * len(bot), height=0.72, zorder=2)
a.set_yticks(range(len(sel))); a.set_yticklabels([t[0] for t in sel], fontsize=5.4)
a.invert_yaxis(); a.set_xlabel("Behavioural elasticity")
xm = max(t[1] for t in sel)
a.set_xlim(0, xm * 1.42)
a.axhline(len(top) - 0.5, color=LGREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
a.text(xm * 1.40, 1.0, "10 stiffest", ha="right", va="center", fontsize=5.8,
       color=BLUE, zorder=4)
a.text(xm * 1.40, 15.0, "10 sloppiest", ha="right", va="center", fontsize=5.8,
       color=GREY, zorder=4)
letter(a, "a", -0.26)

d = L("E1_pyloric_manifold.json")
stiff = [(n, float(v)) for n, v in d["stiff_conductances_lowCV"]]
sloppy = [(n, float(v)) for n, v in d["sloppy_conductances_highCV"]]
sel = stiff + sloppy[::-1]
b = ax[1]
b.barh(range(len(sel)), [t_[1] for t_ in sel],
       color=[BLUE] * len(stiff) + [GREY] * len(sloppy), height=0.72, zorder=2)
b.set_yticks(range(len(sel))); b.set_yticklabels([t_[0] for t_ in sel], fontsize=5.4)
b.invert_yaxis(); b.set_xlabel("Coefficient of variation across the population")
b.set_xlim(0, 1.02)
b.axhline(len(stiff) - 0.5, color=LGREY, lw=0.6, ls=(0, (2, 2)), zorder=1)
b.text(1.0, 1.4, "least variable:\nA-current and\ninhibitory synapses",
       ha="right", va="center", fontsize=5.8, color=BLUE, linespacing=1.25, zorder=4)
b.text(1.0, 7.6, "most variable:\nleak, H and CaT", ha="right", va="center",
       fontsize=5.8, color=GREY, linespacing=1.25, zorder=4)
b.text(1.0, 9.6, f"{d['n_valid']} networks, one rhythm", ha="right", va="center",
       fontsize=6, color=INK, zorder=4)
letter(b, "b", -0.36)
save(fig, "Fig_si_flyvis_pyloric.png")

print("done")
