"""Redraw the two main figures around the connectome-scale measurement.

Figure 2 becomes the BAAIWorm result: the curvature spectrum over 3076 connection
weights, and how that curvature divides between chemical synapses and gap
junctions. Figure 4a becomes the named-conductance ranking measured through the
motor trajectory, and 4b the FlyGym spectrum with its noise floor drawn in, since
that floor is what the figure is really about.

Same house style as the rest of the paper: 183 mm double column, 600 dpi,
Okabe-Ito colours with blue for stiff or identifiable and grey for sloppy.
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
    "legend.fontsize": 6, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "savefig.dpi": 600, "figure.dpi": 600,
})


def panel(ax, letter, dx=-0.16, dy=1.08):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(p).size
    print(f"  {name:36s} {w}x{h}")


# =============================== Figure 2: the connectome-scale measurement ====
b = L("E42_baaiworm_connectome.json")
fig, ax = plt.subplots(1, 2, figsize=(150 * MM, 52 * MM))

# a: the measured distribution of per-connection elasticities. The eigenvalues
# themselves were not retained, so nothing is drawn that was not measured: these
# are the stored percentiles of the per-connection sensitivity.
e90, e99 = b["eff_dim_90"], b["eff_dim_99"]
pr = b["participation_ratio"]
pc = b["elasticity_percentiles"]
q = sorted((float(k), v) for k, v in pc.items())
xs = [t[0] for t in q]
ys = [t[1] for t in q]
ax[0].plot(xs, ys, "o-", color=BLUE, lw=1.3, ms=3.6,
           markeredgecolor="white", markeredgewidth=0.4)
ax[0].set_yscale("log")
ax[0].set_xlabel("Percentile of connections")
ax[0].set_ylabel("Behavioural elasticity (log scale)")
ax[0].set_xlim(0, 100)
NL = chr(10)
ax[0].text(0.03, 0.94,
           f"effective dimension {e90} of 3076 at 90%, {e99} at 99%"
           + NL + f"participation ratio {pr:.2f}",
           transform=ax[0].transAxes, fontsize=6, color=INK, va="top",
           linespacing=1.35)
ax[0].text(0.97, 0.10,
           "elasticities span seven orders" + NL + "of magnitude across connections",
           transform=ax[0].transAxes, ha="right", fontsize=6, color=GREY,
           linespacing=1.3)
panel(ax[0], "a")

# b: how the curvature divides between connection types
cat = b["by_connection_category"]
names, shares, counts = [], [], []
for k, lab in (("syn", "chemical synapse"), ("gj", "gap junction")):
    if k in cat:
        names.append(lab)
        shares.append(cat[k]["share_of_curvature"])
        counts.append(cat[k]["n"])
ypos = np.arange(len(names))
ax[1].barh(ypos, shares, color=[BLUE, GREY][:len(names)], height=0.5)
ax[1].set_yticks(ypos)
ax[1].set_yticklabels([f"{n}\n({c} connections)" for n, c in zip(names, counts)],
                      fontsize=5.8)
ax[1].invert_yaxis()
ax[1].set_xlabel("Share of behavioural curvature")
ax[1].set_xlim(0, 1.0)
for y, sh, c in zip(ypos, shares, counts):
    ax[1].text(sh + 0.02, y, f"{100*sh:.0f}%", va="center", fontsize=6, color=INK)
ax[1].text(0.97, 0.12, "per connection, a chemical synapse\ncarries about five "
                       "times more curvature",
           transform=ax[1].transAxes, ha="right", fontsize=6, color=GREY,
           linespacing=1.3)
panel(ax[1], "b", dx=-0.42)

fig.tight_layout(w_pad=2.6)
save(fig, "Fig_connectome_curvature.png")


# ======================= Figure 4a: named conductances, rich observable ========
c = L("E45_baai_named_channels.json")
rank = c["ranking_stiff_to_sloppy"]
nm = [r[0] for r in rank]
val = np.array([r[1] for r in rank], dtype=float)
WIRING = ("leak", "nca", "irk")
col = [BLUE if any(w in n.lower() for w in WIRING) else GREY for n in nm]

fig, ax = plt.subplots(figsize=(89 * MM, 58 * MM))
y = np.arange(len(val))
ax.hlines(y, val.min() * 0.5, val, color=col, lw=1.0, alpha=0.55)
ax.scatter(val, y, s=12, c=col, zorder=3, edgecolor="white", linewidth=0.3)
ax.set_yticks(y)
ax.set_yticklabels(nm, fontsize=5.4)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("Behavioural elasticity (log scale)")
ax.text(0.97, 0.10, f"effective dimension {c['eff_dim_90']} of "
                    f"{c['n_mechanisms']}\nSLO-1 and SLO-2: no measurable effect",
        transform=ax.transAxes, ha="right", fontsize=6, color=INK, linespacing=1.35)
fig.tight_layout()
ax.text(-0.30, 1.02, "a", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
save(fig, "fig_p2_b1_perchannel.png")


# =================== Figure 4b: FlyGym spectrum with its noise floor ===========
r = L("E1_flygym_rank.json")
nl = L("E1_flygym_null.json")
ev = np.array(r["rich"]["top_eigenvalues"], dtype=float)
ev = np.sort(ev)[::-1]
fig, ax = plt.subplots(figsize=(89 * MM, 58 * MM))
ax.plot(range(1, len(ev) + 1), np.maximum(ev, 1e-12), "o-", color=BLUE, lw=1.0,
        ms=2.6, markeredgecolor="white", markeredgewidth=0.3, label="measured")
ax.axvline(r["rich"]["eff_dim_90"] + 0.5, ls=(0, (3, 3)), color=INK, lw=0.7)
ax.axvline(nl["null_eff_dim_90"] + 0.5, ls=(0, (1, 2)), color=GREY, lw=0.7)
ax.set_yscale("log")
ax.set_xlabel("Eigenvalue index")
ax.set_ylabel("Curvature (log scale)")
ax.text(r["rich"]["eff_dim_90"] + 1.2, ev.max() * 0.30,
        f"{r['rich']['eff_dim_90']} reach 90%", fontsize=6, color=INK)
ax.text(nl["null_eff_dim_90"] - 0.4, ev.min() * 1.25,
        f"noise floor {nl['null_eff_dim_90']}", fontsize=6, color=GREY,
        ha="right", va="bottom")
ax.text(0.02, 0.03, f"{r['rich']['n_observables']} observables against "
                    f"{r['n_params']} parameters",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6, color=INK)
fig.tight_layout()
ax.text(-0.13, 1.02, "b", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
save(fig, "fig_p2_flygym_hessian48.png")

print("done")
