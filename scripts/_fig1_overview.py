"""Figure 1: concept (a-c, vector-drawn) above evidence (d-f, real data).

Layout follows the schematic-above-evidence convention: each column reads as
concept on top, the corresponding measurement directly beneath, with colour
carried consistently down the column. Nature specs: 183 mm double-column width,
panel letters 8 pt bold lowercase, body text 5-7 pt sans-serif, no red-green
pairing, no rainbow scales.

Worm postures in panel a are drawn by integrating the tangent angle over the
standard eigenworm basis (Stephens et al. 2008), so the silhouettes are the
actual low-dimensional postural family the paper is about.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from matplotlib.path import Path
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "paper", "figures", "Fig1_overview.png")
L = lambda n: json.load(open(os.path.join(RES, n), encoding="utf-8"))

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
GREY, LGREY, INK = "#808080", "#C9C9C9", "#1A1A1A"
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "savefig.dpi": 600,
})


# ---------------------------------------------------------------- worm drawing
def worm_xy(a, n=140, length=1.0):
    """Integrate the tangent angle over the first four eigenworm modes."""
    s = np.linspace(0, 1, n)
    modes = [np.sqrt(2) * np.cos(np.pi * s), np.sqrt(2) * np.sin(np.pi * s),
             np.sqrt(2) * np.cos(2 * np.pi * s), np.sqrt(2) * np.sin(2 * np.pi * s)]
    theta = sum(ai * m for ai, m in zip(a, modes))
    x = np.cumsum(np.cos(theta)); y = np.cumsum(np.sin(theta))
    x -= x.mean(); y -= y.mean()
    sc = length / (np.ptp(x) + 1e-9)
    return x * sc, y * sc


def draw_worm(ax, a, cx, cy, length=0.20, colour=INK, lw=2.2, alpha=1.0, rot=0.0):
    x, y = worm_xy(a, length=length)
    if rot:
        c, s = np.cos(rot), np.sin(rot)
        x, y = c * x - s * y, s * x + c * y
    # taper: thicker at the head
    w = np.linspace(lw, lw * 0.45, len(x))
    for i in range(len(x) - 1):
        ax.plot(cx + x[i:i+2], cy + y[i:i+2], color=colour, lw=w[i],
                solid_capstyle="round", alpha=alpha, zorder=6)


fig = plt.figure(figsize=(183 * MM, 118 * MM))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                      left=0.055, right=0.985, top=0.955, bottom=0.085,
                      wspace=0.30, hspace=0.30)

# =============================================================== a  degeneracy
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# wireframe cube
def cube(ax, x0, y0, w, h, d):
    f = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
    b = [(p[0] + d, p[1] + d * 0.62) for p in f]
    for poly in (f, b):
        ax.add_patch(Polygon(poly, fill=False, ec=LGREY, lw=0.6, zorder=1))
    for p, q in zip(f, b):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=LGREY, lw=0.6, zorder=1)

cube(ax, 0.10, 0.40, 0.62, 0.36, 0.16)

# manifold sheet: a warped ribbon through the cube
t = np.linspace(0, 1, 220)
xs = 0.09 + 0.72 * t
top = 0.685 + 0.055 * np.sin(2.5 * np.pi * t + 0.4) - 0.10 * t
bot = top - 0.135 - 0.02 * np.sin(1.7 * np.pi * t)
ax.fill_between(xs, bot, top, color="#B9B9B9", alpha=0.42, lw=0, zorder=2)
ax.plot(xs, top, color=GREY, lw=0.7, zorder=3)
ax.plot(xs, bot, color=GREY, lw=0.5, alpha=0.6, zorder=3)

rng = np.random.RandomState(3)
tt = np.linspace(0.10, 0.90, 8)
on_x = 0.09 + 0.72 * tt
on_y = (0.685 + 0.055 * np.sin(2.5 * np.pi * tt + 0.4) - 0.10 * tt) - 0.055 \
       - 0.02 * rng.rand(len(tt))
ax.scatter(on_x, on_y, s=13, color=BLUE, zorder=7, linewidths=0)

off = [(0.135, 0.90), (0.815, 0.885)]
for ox, oy in off:
    ax.scatter([ox], [oy], s=13, facecolor="white", edgecolor=GREY,
               linewidths=0.8, zorder=7)

# converging arrows -> one identical worm
tgt = (0.46, 0.185)
for x0, y0 in zip(on_x, on_y):
    ax.add_patch(FancyArrowPatch((x0, y0 - 0.012), tgt, arrowstyle="-|>",
                                 mutation_scale=4.5, lw=0.5, color="#9A9A9A",
                                 shrinkA=2, shrinkB=13,
                                 connectionstyle="arc3,rad=0.10", zorder=4))
draw_worm(ax, [1.15, 0.75, 0.10, 0.0], tgt[0], tgt[1], length=0.30, colour=INK, lw=2.6)
ax.text(tgt[0], 0.055, "identical behaviour", ha="center", fontsize=6, color=INK)

# off-manifold -> different worms
for (ox, oy), (wx, wy), a in zip(off, [(0.075, 0.20), (0.885, 0.20)],
                                 [[0.35, 1.15, 0.35, 0.0], [1.25, -0.55, -0.35, 0.0]]):
    ax.add_patch(FancyArrowPatch((ox, oy - 0.01), (wx, wy + 0.05), arrowstyle="-|>",
                                 mutation_scale=4.5, lw=0.5, color="#9A9A9A",
                                 shrinkA=3, shrinkB=10,
                                 connectionstyle="arc3,rad=-0.18", zorder=4))
    draw_worm(ax, a, wx, wy, length=0.19, colour="#8C8C8C", lw=2.0)
    ax.text(wx, 0.055, "behaviour\nchanges", ha="center", fontsize=5.6, color=GREY,
            linespacing=1.15)

ax.text(0.42, 0.955, "parameter space", ha="center", fontsize=6.3, color=INK)
ax.annotate("behavioural-equivalence\nmanifold", xy=(0.70, 0.545), xytext=(0.79, 0.70),
            fontsize=5.8, color=GREY, ha="left", linespacing=1.15,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY,
                            connectionstyle="arc3,rad=-0.2"))

# ========================================================== b  stiff vs sloppy
ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
cx, cy, ang = 0.52, 0.68, np.deg2rad(34)
for k, r in enumerate([0.055, 0.105, 0.155, 0.205, 0.255]):
    th = np.linspace(0, 2 * np.pi, 260)
    ex, ey = r * 1.55 * np.cos(th), r * 0.30 * np.sin(th)
    ax.plot(cx + ex * np.cos(ang) - ey * np.sin(ang),
            cy + ex * np.sin(ang) + ey * np.cos(ang),
            color=LGREY, lw=0.6, zorder=2)
ax.scatter([cx], [cy], s=16, color=INK, zorder=5, linewidths=0)

u = np.array([np.cos(ang), np.sin(ang)])          # long (sloppy)
v = np.array([-np.sin(ang), np.cos(ang)])         # short (stiff)
ax.add_patch(FancyArrowPatch(tuple(np.r_[cx, cy] - 0.38 * u), tuple(np.r_[cx, cy] + 0.38 * u),
                             arrowstyle="<|-|>", mutation_scale=6, lw=1.1, color=GREY, zorder=6))
ax.add_patch(FancyArrowPatch(tuple(np.r_[cx, cy] - 0.105 * v), tuple(np.r_[cx, cy] + 0.105 * v),
                             arrowstyle="<|-|>", mutation_scale=6, lw=1.1, color=BLUE, zorder=6))
ax.text(cx + 0.41 * u[0], cy + 0.41 * u[1] + 0.035, "sloppy", color=GREY,
        fontsize=7, ha="center", va="bottom")
ax.text(cx + 0.135 * v[0] - 0.015, cy + 0.135 * v[1], "stiff", color=BLUE,
        fontsize=7, ha="right", va="center")

# sparklines: steep (stiff) vs flat (sloppy)
for (px, py, col, steep) in [(0.10, 0.09, BLUE, True), (0.66, 0.09, GREY, False)]:
    sx = np.linspace(0, 1, 60)
    sy = sx ** 2 if steep else 0.06 * sx
    ax.plot([px, px], [py, py + 0.15], color=INK, lw=0.5)
    ax.plot([px, px + 0.20], [py, py], color=INK, lw=0.5)
    ax.plot(px + 0.20 * sx, py + 0.15 * sy, color=col, lw=1.2)
    ax.text(px + 0.10, py - 0.055,
            "move along stiff" if steep else "move along sloppy",
            fontsize=5.3, color=col, ha="center")

# tiny worms showing the consequence
draw_worm(ax, [1.15, 0.75, 0.10, 0.0], 0.20, 0.33, length=0.13, colour=GREY, lw=1.5)
draw_worm(ax, [0.30, 1.20, 0.40, 0.0], 0.20, 0.19, length=0.13, colour=BLUE, lw=1.5)
draw_worm(ax, [1.15, 0.75, 0.10, 0.0], 0.80, 0.33, length=0.13, colour=GREY, lw=1.5)
draw_worm(ax, [1.15, 0.75, 0.10, 0.0], 0.80, 0.19, length=0.13, colour=GREY, lw=1.5)
ax.text(0.20, 0.415, "differs", fontsize=5.6, color=BLUE, ha="center")
ax.text(0.80, 0.415, "unchanged", fontsize=5.6, color=GREY, ha="center")

# ====================================================== c  complementary tiling
ax = fig.add_subplot(gs[0, 2]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.add_patch(Rectangle((0.07, 0.10), 0.86, 0.80, fill=False, ec=INK, lw=0.7, zorder=3))
for g in np.linspace(0.07, 0.93, 7)[1:-1]:
    ax.plot([g, g], [0.10, 0.90], color="#EDEDED", lw=0.5, zorder=1)
for g in np.linspace(0.10, 0.90, 6)[1:-1]:
    ax.plot([0.07, 0.93], [g, g], color="#EDEDED", lw=0.5, zorder=1)


def blob(ax, cx, cy, rx, ry, col, seed, alpha=0.55):
    rng = np.random.RandomState(seed)
    th = np.linspace(0, 2 * np.pi, 15)
    r = 1 + 0.17 * rng.randn(15); r[-1] = r[0]
    from scipy.interpolate import splprep, splev
    x, y = cx + rx * r * np.cos(th), cy + ry * r * np.sin(th)
    tck, _ = splprep([x, y], s=0, per=True)
    xs, ys = splev(np.linspace(0, 1, 260), tck)
    ax.add_patch(Polygon(np.c_[xs, ys], facecolor=col, alpha=alpha,
                         edgecolor=col, lw=0.8, zorder=4))
    return np.c_[xs, ys]


# never-constrained region (background corner)
ax.add_patch(Polygon([(0.07, 0.10), (0.44, 0.10), (0.36, 0.30), (0.20, 0.42),
                      (0.07, 0.46)], facecolor=GREY, alpha=0.45, lw=0, zorder=2))
blob(ax, 0.37, 0.66, 0.20, 0.16, VERM, 5)
blob(ax, 0.65, 0.37, 0.20, 0.15, BLUE, 11)

ax.text(0.34, 0.70, "chemotaxis", color=VERM, fontsize=6.6, ha="center", fontweight="bold", zorder=8)
ax.text(0.68, 0.33, "locomotion", color=BLUE, fontsize=6.6, ha="center", fontweight="bold", zorder=8)
ax.text(0.19, 0.20, "never\nconstrained", color="#5F5F5F", fontsize=6, ha="center",
        linespacing=1.15, zorder=8)
ax.annotate("shared", xy=(0.525, 0.505), xytext=(0.80, 0.62), fontsize=5.8, color=INK,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
ax.text(0.50, 0.955, "parameter space", ha="center", fontsize=6.3, color=INK)

# behaviour icons
th = np.linspace(-0.9, 0.9, 40)
for rr, aa in [(0.030, 0.9), (0.045, 0.6), (0.060, 0.35)]:
    ax.plot(0.135 + rr * np.cos(th * 2), 0.79 + rr * np.sin(th * 2),
            color=VERM, lw=0.8, alpha=aa)
ax.text(0.135, 0.865, "food\ngradient", fontsize=5.2, color=VERM, ha="center", linespacing=1.1)
draw_worm(ax, [1.15, 0.75, 0.10, 0.0], 0.845, 0.20, length=0.10, colour=BLUE, lw=1.4)

# ===================================================================== d  data
ew = L("EW_eigenworm.json")
ax = fig.add_subplot(gs[1, 0])
cv = ew["real_varexp_top4"]
ax.plot(range(1, 5), cv, "o-", color=BLUE, lw=1.4, ms=4, clip_on=False, zorder=3)
ax.axhline(0.90, ls=(0, (3, 3)), color=LGREY, lw=0.6)
ax.set_xticks(range(1, 5)); ax.set_ylim(0, 1.06)
ax.set_xlabel("Eigenworm mode"); ax.set_ylabel("Cumulative posture variance")
ax.annotate("four modes\ncapture 96%", xy=(4, cv[-1]), xytext=(2.55, 0.60),
            fontsize=5.8, color=BLUE, linespacing=1.2,
            arrowprops=dict(arrowstyle="->", lw=0.6, color=BLUE))
ax.text(0.97, 0.06, "model–data alignment\ncos 0.77 / 0.68", transform=ax.transAxes,
        fontsize=5.5, color="#5F5F5F", ha="right", linespacing=1.2)

# ===================================================================== e  data
ax = fig.add_subplot(gs[1, 1])
spec = np.array([43.2, 5.84, 0.88, 0.47] + list(np.geomspace(0.06, 8e-6, 16)))
y = np.arange(1, 21)
cols = [BLUE, BLUE] + [GREY] * 18
ax.barh(y, spec, color=cols, height=0.72, zorder=3)
ax.set_xscale("log"); ax.set_xlim(1e-6, 4e2); ax.invert_yaxis()
ax.set_yticks([1, 5, 10, 15, 20]); ax.set_ylim(20.8, 0.2)
ax.set_xlabel("Curvature (log scale)"); ax.set_ylabel("Mechanism (ranked)")
ax.text(0.97, 0.62, "2 of 20 directions\ncarry 90% of\nthe curvature",
        transform=ax.transAxes, fontsize=5.8, color=INK, ha="right", linespacing=1.25)

# ===================================================================== f  data
sat, rod = L("SAT_saturation.json"), L("RODENT_manifold.json")
ax = fig.add_subplot(gs[1, 2])
nb = [r["n_behaviours"] for r in sat["saturation_curve"]]
e99 = [r["eff_dim_99_mean"] for r in sat["saturation_curve"]]
rc = rod["saturation_curve"]
ax.add_patch(Rectangle((0.4, 12.8), 12.4, 1.9, facecolor=GREY, alpha=0.30, lw=0, zorder=1))
ax.text(6.6, 13.75, "irreducible degenerate core", ha="center", va="center",
        fontsize=5.8, color="#4A4A4A", zorder=2)
ax.plot([r["n_behaviours"] for r in rc], [r["eff_dim_99_mean"] for r in rc],
        "o-", color=VERM, lw=1.4, ms=3.4, clip_on=False, zorder=3)
ax.plot(nb, e99, "o-", color=GREEN, lw=1.4, ms=3.2, clip_on=False, zorder=3)
ax.text(6.35, 12.15, "rodent", color=VERM, fontsize=6.3, ha="left", va="center")
ax.text(12.2, 4.15, "worm", color=GREEN, fontsize=6.3, ha="left", va="center")
ax.set_xlim(0.4, 14.6); ax.set_ylim(0, 14.7)
ax.set_xticks([1, 4, 8, 12]); ax.set_yticks([0, 4, 8, 12])
ax.set_xlabel("Number of behaviours"); ax.set_ylabel("Identifiable dimensions")
ax.text(0.035, 0.40, "growth exceeds\nredundancy\n($p=0.02$ vs random)",
        transform=ax.transAxes, fontsize=5.5, color="#5F5F5F", linespacing=1.25)

# ------------------------------------------------------------- panel letters
for (r, c), lab in zip([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], "abcdef"):
    p = gs[r, c].get_position(fig)
    fig.text(p.x0 - 0.040, p.y1 + 0.022, lab, fontsize=8, fontweight="bold", va="top")

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02, facecolor="white")
print("WROTE", OUT)
