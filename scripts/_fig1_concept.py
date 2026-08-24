"""Figure 1: what behaviour determines about a nervous-system model.

Replaces the externally generated bitmap with a drawn schematic, so the figure
carries the same numbers as the results files and is legible at print size.

Top     the many-to-one map from parameters to behaviour, and its anisotropy
Middle  the three consequences: ground truth, biophysical partition, repertoire
Bottom  the measurement, and the three gates a reported dimension must pass

Every element is placed in figure coordinates from the band table below, so
that no label can drift into a neighbouring panel when the numbers change.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse, Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "paper", "figures")

BLUE, GREY, VERM, INK = "#0072B2", "#808080", "#D55E00", "#1A1A1A"
LGREY, PALE = "#C9C9C9", "#EDEDED"
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 1.8, "ytick.major.size": 1.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
    "savefig.dpi": 600, "figure.dpi": 600,
})

fig = plt.figure(figsize=(180 * MM, 152 * MM))

BANDS = ((0.978, "Behaviour is a many-to-one map, and the manifold it leaves is extremely anisotropic"),
         (0.638, "Three consequences"),
         (0.298, "The measurement, and the three gates a reported dimension must pass"))
for y, t in BANDS:
    fig.text(0.012, y, t, fontsize=7.4, fontweight="bold", va="center")


def bare(rect):
    ax = fig.add_axes(rect)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.patch.set_alpha(0)
    return ax


def arrow(ax, a, b, col=INK, lw=0.9, ms=6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=col, shrinkA=1, shrinkB=1, clip_on=False))


# ============================== TOP BAND: 0.700 - 0.940 ==============================
axA = bare([0.030, 0.700, 0.245, 0.230])
axA.text(0.5, 1.03, "parameter space", ha="center", fontsize=6.8, fontweight="bold")

rng = np.random.default_rng(3)
ang = np.deg2rad(26)
R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
axA.add_patch(Ellipse((0.5, 0.50), 0.66, 0.15, angle=np.degrees(ang),
                      fc=PALE, ec=LGREY, lw=0.6, zorder=1))
t = rng.uniform(-1, 1, 130)
pts = (R @ np.vstack([t * 0.30, rng.normal(0, 0.020, 130)])).T + [0.5, 0.50]
axA.plot(pts[:, 0], pts[:, 1], "o", ms=1.5, color=GREY, mew=0, zorder=2)
axA.plot(*((R @ np.array([-0.20, 0.0])) + [0.5, 0.50]), "o", ms=3.6,
         color=BLUE, mec="white", mew=0.5, zorder=3)

stiff = R @ np.array([0.0, 0.105])
axA.annotate("", xy=(0.5 + stiff[0], 0.50 + stiff[1]), xytext=(0.5 - stiff[0], 0.50 - stiff[1]),
             arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color=BLUE, mutation_scale=5))
axA.text(0.735, 0.855, "determined: 8 directions", color=BLUE, fontsize=6, ha="right")
arrow(axA, (0.660, 0.825), (0.560, 0.605), col=BLUE, lw=0.5, ms=4)
axA.text(0.265, 0.155, "free: 3068 directions", color=GREY, fontsize=6, ha="left")
arrow(axA, (0.365, 0.195), (0.430, 0.375), col=GREY, lw=0.5, ms=4)
axA.text(0.5, 0.015, "3076 connectome weights", ha="center", fontsize=6)

axM = bare([0.278, 0.700, 0.056, 0.230])
arrow(axM, (0.10, 0.52), (0.92, 0.52))
axM.text(0.51, 0.60, "simulate", ha="center", fontsize=6)

axB = bare([0.340, 0.700, 0.200, 0.230])
axB.text(0.5, 1.03, "behaviour", ha="center", fontsize=6.8, fontweight="bold")
x = np.linspace(0, 1, 300)
for dy, col, lw in ((0.72, BLUE, 1.1), (0.55, GREY, 0.8), (0.38, GREY, 0.8)):
    axB.plot(0.10 + 0.80 * x, dy + 0.070 * np.sin(2 * np.pi * 1.6 * x), color=col, lw=lw)
axB.text(0.5, 0.175, "one and the same behaviour, from", ha="center", fontsize=6)
axB.text(0.5, 0.085, "every parameter set on the manifold", ha="center", fontsize=6, color=GREY)

axC = bare([0.590, 0.700, 0.390, 0.230])
axC.text(0.0, 1.03, "a few per cent, in two phyla", fontsize=6.8, fontweight="bold")
for i, (name, k, n) in enumerate((("nematode connectome model", 8, 3076),
                                  ("fly optic-lobe model", 14, 330))):
    y = 0.545 - i * 0.315
    axC.text(0.0, y + 0.175, name, fontsize=6)
    axC.add_patch(Rectangle((0.0, y), 0.58, 0.105, fc=PALE, ec=LGREY, lw=0.5))
    axC.add_patch(Rectangle((0.0, y), 0.58 * max(k / n, 0.010), 0.105, fc=BLUE, ec="none"))
    axC.text(0.605, y + 0.053, f"{k} of {n}", fontsize=6, va="center",
             color=BLUE, fontweight="bold")
    axC.text(0.855, y + 0.053, f"{100*k/n:.1f}%", fontsize=6, va="center", color=GREY)
axC.text(0.0, 0.045, "fraction of the parameter space behaviour determines",
         fontsize=6, color=GREY)

# ============================= MIDDLE BAND: 0.360 - 0.560 =============================
MY, MH = 0.360, 0.200

ax1 = fig.add_axes([0.060, MY, 0.215, MH])
ax1.set_title("a fit can match behaviour\nwithout recovering parameters",
              fontsize=6.5, pad=4, linespacing=1.3)
ax1.plot([0.02, 0.55], [0.02, 0.55], color=LGREY, lw=0.7, ls=(0, (3, 3)), zorder=1)
ax1.plot(0.001, 0.001, "o", ms=4, color=BLUE, mec="white", mew=0.5, zorder=3)
ax1.plot(0.001, 0.40, "o", ms=4, color=VERM, mec="white", mew=0.5, zorder=3)
ax1.annotate("2-4-parameter\nneuron models", (0.001, 0.001), (0.115, 0.075), fontsize=5.8,
             color=BLUE, linespacing=1.25,
             arrowprops=dict(arrowstyle="-", lw=0.5, color=BLUE))
ax1.annotate("MAPK cascade", (0.001, 0.40), (0.115, 0.435), fontsize=5.8, color=VERM,
             arrowprops=dict(arrowstyle="-", lw=0.5, color=VERM))
ax1.set_xlim(-0.03, 0.58), ax1.set_ylim(-0.04, 0.58)
ax1.set_xlabel("behavioural distance", labelpad=1.5)
ax1.set_ylabel("parameter distance from truth", labelpad=1.5)
ax1.set_xticks([0, 0.25, 0.5]), ax1.set_yticks([0, 0.25, 0.5])

ax2 = fig.add_axes([0.400, MY, 0.195, MH])
ax2.set_title("what is determined is\nbiophysically specific", fontsize=6.5, pad=4, linespacing=1.3)
bars = [("chemical\nsynapses", 87, BLUE, "87%"), ("gap\njunctions", 13, BLUE, "13%"),
        ("fast Ca$^{2+}$\nconductances", 2, GREY, "free")]
for i, (lab, v, col, txt) in enumerate(bars):
    ax2.bar(i, v, 0.58, color=col, ec="none")
    ax2.text(i, v + 4, txt, ha="center", fontsize=6, color=col, fontweight="bold")
ax2.set_xticks(range(3)), ax2.set_xticklabels([b[0] for b in bars], fontsize=5.8, linespacing=1.2)
ax2.set_ylim(0, 108), ax2.set_yticks([0, 50, 100])
ax2.set_ylabel("share of the curvature", labelpad=1.5)

ax3 = bare([0.690, MY, 0.290, MH])
ax3.text(0.5, 1.115, "the identified set depends on", ha="center", fontsize=6.5)
ax3.text(0.5, 1.020, "which behaviour is elicited", ha="center", fontsize=6.5)
ax3.text(0.175, 0.815, "locomotion", ha="center", fontsize=6, color=BLUE)
ax3.text(0.680, 0.815, "chemotaxis", ha="center", fontsize=6, color=VERM)
ax3.add_patch(Ellipse((0.320, 0.545), 0.42, 0.36, fc=BLUE, ec="none", alpha=0.30))
ax3.add_patch(Ellipse((0.540, 0.545), 0.42, 0.36, fc=VERM, ec="none", alpha=0.30))
ax3.text(0.155, 0.545, "4", ha="center", va="center", fontsize=7.5, color=BLUE, fontweight="bold")
ax3.text(0.705, 0.545, "4", ha="center", va="center", fontsize=7.5, color=VERM, fontweight="bold")
ax3.text(0.430, 0.545, "2", ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold")
ax3.text(0.430, 0.245, "together: 6 directions", ha="center", fontsize=6.2, fontweight="bold")
ax3.text(0.430, 0.115, "more of one action re-measures what", ha="center", fontsize=5.8, color=GREY)
ax3.text(0.430, 0.020, "it already determines (cos 0.54, null 0.31)",
         ha="center", fontsize=5.8, color=GREY)

# ============================= BOTTOM BAND: 0.030 - 0.265 =============================
# the pipeline boxes are placed in figure coordinates so their corners stay
# circular: a rounded box drawn in a wide, short axes comes out as a lozenge
BOX_W, BOX_H, BOX_Y = 0.146, 0.050, 0.205
for lab, cx in (("perturb\neach parameter", 0.103), ("re-simulate\nbehaviour", 0.273),
                ("record the change\nin the observables", 0.443), ("assemble $J$", 0.613)):
    fig.add_artist(FancyBboxPatch((cx - BOX_W / 2, BOX_Y), BOX_W, BOX_H,
                                  boxstyle="round,pad=0.004,rounding_size=0.010",
                                  fc="white", ec=GREY, lw=0.6,
                                  transform=fig.transFigure, zorder=2))
    fig.text(cx, BOX_Y + BOX_H / 2, lab, ha="center", va="center",
             fontsize=6, linespacing=1.25, zorder=3)
axP = bare([0.0, 0.0, 1.0, 1.0])
axP.set_zorder(1)
for cx in (0.103, 0.273, 0.443):
    arrow(axP, (cx + BOX_W / 2 + 0.005, BOX_Y + BOX_H / 2),
          (cx + 0.170 - BOX_W / 2 - 0.005, BOX_Y + BOX_H / 2), col=GREY, lw=0.7, ms=5)

axJ = fig.add_axes([0.062, 0.052, 0.105, 0.108])
J = np.abs(rng.normal(0, 1, (34, 13))) ** 2.2
J[:, :2] *= 9
axJ.imshow(J, aspect="auto", cmap="Blues", vmax=np.percentile(J, 99))
axJ.set_xticks([]), axJ.set_yticks([])
for s in axJ.spines.values():
    s.set_visible(True), s.set_lw(0.5)
axJ.set_ylabel("observables", fontsize=5.8, labelpad=2)
axJ.set_xlabel("parameters", fontsize=5.8, labelpad=2)

axH = bare([0.180, 0.052, 0.185, 0.108])
arrow(axH, (0.03, 0.55), (0.24, 0.55), col=GREY, lw=0.7, ms=5)
axH.text(0.50, 0.58, r"$H = J^{\top} W J$", ha="center", va="center", fontsize=8)
axH.text(0.50, 0.32, "diagonalise", ha="center", va="center", fontsize=6, color=GREY)
arrow(axH, (0.76, 0.55), (0.97, 0.55), col=GREY, lw=0.7, ms=5)

axS = fig.add_axes([0.400, 0.052, 0.165, 0.108])
ev = np.array([1.0, 0.42, 0.11] + list(10 ** np.linspace(-1.4, -6.2, 17)))
axS.plot(range(1, len(ev) + 1), ev, "o-", ms=2.0, lw=0.8, color=GREY, mec="white", mew=0.3, zorder=2)
axS.plot(range(1, 4), ev[:3], "o", ms=3.0, color=BLUE, mec="white", mew=0.4, zorder=3)
axS.set_yscale("log")
axS.set_xlabel("eigenvalue rank", labelpad=1.5)
axS.set_ylabel("curvature", labelpad=1.5)
axS.set_yticks([1e0, 1e-3, 1e-6])
axS.set_xlim(-1, 21)
axS.text(4.6, 0.34, "determined", fontsize=5.8, color=BLUE)
axS.text(11.0, 8e-4, "free", fontsize=5.8, color=GREY)

axG = bare([0.700, 0.030, 0.280, 0.238])
axG.text(0.0, 0.955, "reported only if", fontsize=6.5, fontweight="bold", va="center")
for i, g in enumerate(("observables outnumber parameters",
                       "response proportional to the perturbation",
                       "signal above the simulator's own noise")):
    y = 0.700 - i * 0.185
    axG.plot(0.035, y, "o", ms=2.6, color=BLUE, mec="none", clip_on=False)
    axG.text(0.090, y, g, fontsize=6, va="center")
axG.text(0.0, 0.075, "dimensions failing any of the three", fontsize=5.8, color=GREY, va="center")
axG.text(0.0, -0.010, "are withdrawn", fontsize=5.8, color=GREY, va="center")

p = os.path.join(FIG, "Fig1_concept.png")
fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
fig.savefig(p.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)

from PIL import Image
w, h = Image.open(p).size
print(f"Fig1_concept.png  {w}x{h}  = {w/300*25.4:.0f} mm at 300 dpi "
      f"({w/(0.98*170/25.4):.0f} dpi as placed)")
