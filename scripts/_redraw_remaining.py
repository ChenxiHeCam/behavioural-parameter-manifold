"""Redraw every remaining main-text figure in the house style.

The figures rebuilt here were still carrying matplotlib defaults: boxed axes,
in-figure titles, legend boxes, arbitrary hues (brown, purple, red) and, in two
cases, annotation text sitting on top of the bars it described. Two of them also
lacked the panel letters their captions refer to.

House style, identical to the figures already redrawn:
  blue    #0072B2  stiff / identifiable
  grey    #808080  sloppy / unconstrained
  vermil. #D55E00  a second system or condition
  green   #009E73  a third system or condition
183 mm double column or 89 mm single, 600 dpi, 5-7 pt sans-serif, left and bottom
spines only, ticks outward, no gridlines, no red, no in-figure titles.
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


def panel(ax, letter, dx=-0.20, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(p).size
    print(f"  {name:36s} {w}x{h}")


# ===================================================== Fig 2a  multistart drift
mw = L("modworm_REAL_manifold.json")
fg = L("flygym_hessian48.json")

fig, ax = plt.subplots(1, 3, figsize=(183 * MM, 50 * MM))

# left: parameter perturbation against behavioural change, real modWorm
v = [mw["param_relerr_mean"], mw["behav_dist_mean"]]
ax[0].bar([0, 1], v, color=[GREY, BLUE], width=0.42)
ax[0].set_yscale("log")
ax[0].set_xticks([0, 1])
ax[0].set_xticklabels(["parameter\nperturbation", "behavioural\nchange"])
ax[0].set_xlim(-0.7, 1.7)
ax[0].set_ylabel("Normalised dispersion")
ax[0].set_ylim(3e-3, 1.4)
ax[0].tick_params(axis="x", length=0)
for x, y in zip([0, 1], v):
    ax[0].text(x, y * 1.22, f"{y:.3f}", ha="center", fontsize=5.8, color=INK)
r = mw["behav_over_param_ratio"]
ax[0].text(1.62, 0.55, f"behaviour\n{1/r:.0f}$\\times$ tighter\nthan the\nparameters",
           ha="right", va="top", fontsize=5.8, color=INK, linespacing=1.35,
           transform=ax[0].get_xaxis_transform())
panel(ax[0], "a", dx=-0.26)

# right: held-out behavioural distance before and after recovery, FlyGym
init, rec, sd = 2.80e-4, 6.4e-5, 2.5e-5
ax[1].bar([0, 1], [init, rec], color=[GREY, BLUE], width=0.42)
ax[1].errorbar([1], [rec], yerr=[sd], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.6)
ax[1].set_yscale("log")
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["independent\nstarts", "recovered"])
ax[1].set_xlim(-0.7, 1.7)
ax[1].set_ylim(2.5e-5, 6.5e-4)
ax[1].set_ylabel("Held-out behavioural distance")
ax[1].tick_params(axis="x", length=0)
ax[1].text(1.62, 0.97, "29 of 30 runs\nreach the target;\nthe solution cloud\nstill occupies\n"
                       "21.7 of 126\ndimensions",
           ha="right", va="top", fontsize=5.8, color=INK, linespacing=1.35,
           transform=ax[1].get_xaxis_transform())
panel(ax[1], "b", dx=-0.26)

# right: Gauss-Newton curvature per mechanism, real modWorm
mb = L("modworm_REAL_b1.json")
mech = mb["mechanisms"]
el = mb.get("per_mechanism_elasticity") or {}
mv = ([abs(float(el[m])) for m in mech] if isinstance(el, dict) and el
      else [abs(x) for x in mb["eigenvalues"]])
nm = mb.get("names") if isinstance(mb.get("names"), dict) else {}
order = np.argsort(mv)[::-1]
mlab = [nm.get(mech[i], mech[i]) for i in order]
mval = np.array(mv)[order]
# grey the axes that sit at the finite-difference floor, orders below the rest;
# the 90% cut of the eigen-spectrum is a different quantity and is stated in the
# caption rather than drawn here, so the two cannot disagree on the page
mcol = [GREY if v < 1e-3 * mval.max() else BLUE for v in mval]
ax[2].barh(range(len(mval)), mval, color=mcol, height=0.62)
ax[2].set_yticks(range(len(mval)))
ax[2].set_yticklabels(mlab, fontsize=5.2)
ax[2].invert_yaxis()
ax[2].set_xscale("log")
ax[2].set_xlabel("Behavioural elasticity (log scale)")
ax[2].text(0.97, 0.08, "blue: carries the curvature\ngrey: behaviourally flat",
           transform=ax[2].transAxes, ha="right", fontsize=5.5, color=GREY, linespacing=1.3)
panel(ax[2], "c", dx=-0.44)

fig.tight_layout(w_pad=2.6)
save(fig, "Fig_multistart_three_sim.png")


# ============================================ Fig 4a  BAAIWorm named mechanisms
d = L("baai_perchannel_hessian.json")
el = d.get("per_channel_stiffness") or {}
gn = d.get("gene_names", {})
items = sorted(((gn.get(k, k), abs(float(v))) for k, v in el.items()),
               key=lambda t: t[1], reverse=True)
names = [t[0] for t in items]
vals = np.array([t[1] for t in items])
WIRING = ("gap", "synap", "motor", "leak", "passive", "nca")
cls = [BLUE if any(w in n.lower() for w in WIRING) else GREY for n in names]

# a wide, short lollipop reads far better at 20 rows than a tall bar stack
fig, ax = plt.subplots(figsize=(89 * MM, 62 * MM))
y = np.arange(len(vals))
ax.hlines(y, 1e-3, vals, color=cls, lw=1.0, alpha=0.55)
ax.scatter(vals, y, s=11, c=cls, zorder=3, edgecolor="white", linewidth=0.3)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=4.8)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlim(6e-3, 12)
ax.set_xlabel("Behavioural elasticity (log scale)")
ax.set_ylim(len(vals) - 0.3, -1.9)
ax.text(6.5e-3, -1.5, "wiring and passive", fontsize=5.5, color=BLUE, va="center")
ax.text(0.42, -1.5, "single-cell channels", fontsize=5.5, color=GREY, va="center")
fig.tight_layout()
save(fig, "fig_p2_b1_perchannel.png")


# ================================== Fig 5  four biological datasets, no simulator
e1 = L(os.path.join("e1b_results", "e1b_hessian.json"))
e2 = L("E2_allen_variability.json")
e4 = L("E4_channel_conservation.json")
e5 = L("E5_n2_manifold.json")

fig, ax = plt.subplots(2, 2, figsize=(183 * MM, 108 * MM))

# a  pyloric Hessian scree
spec = np.array(e1["mean_spectrum_sorted"], dtype=float)
spec = np.abs(spec)
for s in e1["spectrum_per_point"]:
    s = np.sort(np.abs(np.array(s, dtype=float)))[::-1]
    ax[0, 0].plot(range(1, len(s) + 1), np.maximum(s, 1e-6), color=LGREY, lw=0.4, zorder=1)
ax[0, 0].plot(range(1, len(spec) + 1), np.maximum(spec, 1e-6), "o-", color=BLUE,
              lw=1.0, ms=2.2, markeredgecolor="white", markeredgewidth=0.3, zorder=3)
ax[0, 0].set_yscale("log")
ax[0, 0].set_xlabel("Eigenvalue index")
ax[0, 0].set_ylabel("Curvature (log scale)")
ax[0, 0].text(0.97, 0.93, f"effective dimension {e1['eff_dim_mean']:.2f} of 31\n"
                          f"across {e1['mean_spectrum_range_oom']:.0f} orders of magnitude",
              transform=ax[0, 0].transAxes, ha="right", va="top", fontsize=6,
              color=INK, linespacing=1.3)
ax[0, 0].text(0.97, 0.72, f"grey: each of the {e1['K']} valid points\nblue: mean",
              transform=ax[0, 0].transAxes, ha="right", va="top", fontsize=5.5, color=GREY,
              linespacing=1.3)
panel(ax[0, 0], "a", dx=-0.16)

# b  Allen cell-types variability
cv = sorted(e2["cv_per_feature"].items(), key=lambda t: t[1])
lab = [k.replace("ef__", "").replace("_long_square", "").replace("_", " ") for k, _ in cv]
val = [v for _, v in cv]
col = [BLUE if v < 0.3 else GREY for v in val]
ax[0, 1].barh(range(len(val)), val, color=col, height=0.68)
ax[0, 1].set_yticks(range(len(val)))
ax[0, 1].set_yticklabels(lab, fontsize=5.2)
ax[0, 1].set_xlabel("Coefficient of variation across neurons")
ax[0, 1].text(0.97, 0.20, f"participation dimension {e2['participation_eff_dim']:.2f} of 11\n"
                          f"n = {e2['n_cells']:,} neurons",
              transform=ax[0, 1].transAxes, ha="right", fontsize=6, color=INK, linespacing=1.3)
ax[0, 1].text(0.97, 0.06, "blue: least variable      grey: most variable",
              transform=ax[0, 1].transAxes, ha="right", fontsize=5.5, color=GREY)
panel(ax[0, 1], "b", dx=-0.30)

# c  ortholog conservation
rank = e4["rank_most_to_least_conserved"]
pid = np.array([r["mean_perc_id"] for r in rank], dtype=float)
ax[1, 0].hist(pid, bins=np.arange(76, 101, 2), color=LGREY, edgecolor="white", linewidth=0.4)
top = rank[0]
ax[1, 0].axvline(top["mean_perc_id"], color=BLUE, lw=0.9)
ax[1, 0].text(top["mean_perc_id"] - 0.7, ax[1, 0].get_ylim()[1] * 0.93,
              f"{top['gene']}  {top['mean_perc_id']:.1f}%", ha="right", fontsize=6, color=BLUE)
ax[1, 0].axvline(e4["conservation_distribution_perc_id"]["median"], color=INK,
                 lw=0.7, ls=(0, (3, 3)))
ax[1, 0].text(e4["conservation_distribution_perc_id"]["median"] - 0.7,
              ax[1, 0].get_ylim()[1] * 0.60,
              f"median {e4['conservation_distribution_perc_id']['median']:.1f}%",
              ha="right", fontsize=5.8, color=INK)
ax[1, 0].set_xlabel("Ortholog identity across $\\it{Caenorhabditis}$ (%)")
ax[1, 0].set_ylabel("Number of genes")
ax[1, 0].text(0.03, 0.93, f"n = {e4['conservation_distribution_perc_id']['n_genes_scored']} "
                          "channel and synaptic genes",
              transform=ax[1, 0].transAxes, fontsize=6, color=INK, va="top")
panel(ax[1, 0], "c", dx=-0.16)

# d  isogenic N2 behavioural manifold
vr = np.array(e5["pca_var_ratio"], dtype=float)
cum = np.cumsum(vr) / vr.sum()
ax[1, 1].plot(range(1, len(cum) + 1), cum, "-", color=GREEN, lw=1.2)
for lev, c in ((0.90, INK), (0.99, LGREY)):
    ax[1, 1].axhline(lev, color=c, lw=0.6, ls=(0, (3, 3)))
k90, k99 = e5["eff_dim_90"], e5["eff_dim_99"]
ax[1, 1].plot([k90], [cum[k90 - 1]], "o", color=GREEN, ms=3.2,
              markeredgecolor="white", markeredgewidth=0.4)
ax[1, 1].annotate(f"{k90} components reach 90%", (k90, cum[k90 - 1]),
                  textcoords="offset points", xytext=(6, -9), fontsize=6, color=INK)
ax[1, 1].plot([k99], [cum[k99 - 1]], "o", color=GREY, ms=3.0,
              markeredgecolor="white", markeredgewidth=0.4)
ax[1, 1].annotate(f"{k99} reach 99%", (k99, cum[k99 - 1]),
                  textcoords="offset points", xytext=(4, -22), fontsize=6, color=GREY)
ax[1, 1].set_xlabel("Principal component")
ax[1, 1].set_ylabel("Cumulative variance")
ax[1, 1].set_ylim(0, 1.05)
ax[1, 1].text(0.97, 0.35, f"participation dimension {e5['participation_eff_dim']:.2f}\n"
                          f"n = {e5['n_worms']} isogenic animals, "
                          f"{e5['n_features']} features",
              transform=ax[1, 1].transAxes, ha="right", fontsize=6, color=INK, linespacing=1.3)
panel(ax[1, 1], "d", dx=-0.16)

fig.tight_layout(w_pad=3.4, h_pad=3.0)
save(fig, "Fig_biological_datasets.png")

print("done")
