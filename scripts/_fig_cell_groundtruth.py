"""SI figure: what ground truth adds at the cell scale.

Panel a  The degeneracy itself, in the model's own output: the MAPK oscillation
         at the true parameters, at a fitted set 43% away whose individual rate
         constants differ by up to eight-fold yet whose behaviour is matched to
         3%, and at a displacement of the same magnitude taken off the
         equivalence manifold, which changes the rhythm outright.

Panel b  Behaviour distance against parameter error for every converged fit in
         the protocol-matched recovery run (11 cell models, random starts drawn
         independently of the truth). The identifiable models sit at the origin:
         behaviour matched, parameters exact. The degenerate models sit up the
         left edge: behaviour matched, parameters wrong -- direct evidence that
         a perfect behavioural fit does not imply recovered parameters.

Panel c  The same Hodgkin-Huxley model measured twelve ways. The 90% effective
         dimension is 1 in every cell of the parameter-count x observable grid,
         while the actually-achieved recovery error spans 4% to 56%: the
         effective dimension is a statement about curvature concentration, not
         an upper bound on what recovery achieves.

Style follows the manuscript: Okabe-Ito, blue #0072B2 = identifiable/determined,
grey #808080 = degenerate/free, vermillion #D55E00 = second condition, 89 mm
single column, 600 dpi, 5-7 pt sans-serif.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "paper", "figures")

BLUE, GREY, VERM, INK, LGREY = "#0072B2", "#808080", "#D55E00", "#1A1A1A", "#C9C9C9"
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

# ---------------------------------------------------------------- panel a data
d3 = json.load(open(os.path.join(W, "cell_panel3_result.json")))["by_model"]
d3b = json.load(open(os.path.join(W, "cell_panel3b_result.json")))["by_model"]
merged = dict(d3)
merged.update(d3b)          # the aliasing-corrected rerun supersedes those models

pts = []                    # (param error after fit, n_params, converged fraction)
for name, m in merged.items():
    if m.get("rel_after_median") is None:
        continue
    pts.append((name, m["n_params"], m["rel_after_median"],
                m["convergence_rate"], m["class"]))

show = json.load(open(os.path.join(W, "mapk_showcase.json")))
tr = {k: np.asarray(v) for k, v in show["traces"].items()}
t = np.linspace(0, 6000.0, len(tr["truth"]))
w = (t >= 3000)                       # the settled portion of the oscillation

fig = plt.figure(figsize=(183 * MM, 62 * MM))
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.05, 1.0], hspace=0.35,
                      wspace=0.42)
axT = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[1, 0], sharex=axT)
ax = [fig.add_subplot(gs[:, 1]), fig.add_subplot(gs[:, 2])]

rel = show["best_fit"]["rel_after"]
axT.plot(t[w], tr["truth"][w], color=INK, lw=0.9)
axT.plot(t[w], tr["recovered"][w], color=BLUE, lw=0.9, ls=(0, (4, 2)))
axT.text(0.0, 1.05, f"fitted, parameters {rel*100:.0f}% away",
         transform=axT.transAxes, fontsize=5.6, color=BLUE, va="bottom")
axB.plot(t[w], tr["truth"][w], color=INK, lw=0.9)
axB.plot(t[w], tr["random"][w], color=VERM, lw=0.9)
axB.text(0.0, 1.05, "same displacement, off the manifold",
         transform=axB.transAxes, fontsize=5.6, color=VERM, va="bottom")
for a_ in (axT, axB):
    a_.set_ylabel("MAPK-PP", fontsize=6)
    a_.tick_params(labelsize=5.2)
axT.tick_params(labelbottom=False)
axB.set_xlabel("Time (s)")
axT.text(-0.30, 1.12, "a", transform=axT.transAxes, fontsize=8,
         fontweight="bold", va="bottom")

a = ax[0]
for name, npar, err, conv, cls in pts:
    col = BLUE if err < 0.02 else GREY
    a.scatter(npar, max(err, 4e-4), s=22 * max(conv, 0.15), color=col,
              edgecolor="white", linewidth=0.3, zorder=3)
a.set_yscale("log")
a.set_xlabel("Number of parameters")
a.set_ylabel("Parameter error")
a.set_xticks([2, 4, 6, 8, 10])
a.set_ylim(2.5e-4, 1.2)
a.axhline(4e-4, color=LGREY, lw=0.6, ls=(0, (3, 3)), zorder=1)
a.text(2.0, 5.5e-4, "recovered exactly", fontsize=5.6, color=BLUE, va="bottom")
lab = {"MAPK": (11, 0.399), "Goodwin": (8, 0.097), "GoldbeterMitotic": (9, 0.166),
       "Repressilator": (4, 0.137)}
for k, (x, y) in lab.items():
    a.annotate(k if k != "GoldbeterMitotic" else "Goldbeter",
               (x, y), textcoords="offset points", xytext=(-4, 5), ha="right",
               fontsize=5.2, color=INK)
a.annotate("behaviour matched,\nparameters wrong", (7.6, 0.62), ha="right",
           fontsize=5.8, color=GREY, linespacing=1.3)
a.text(-0.13, 1.02, "b", transform=a.transAxes, fontsize=8, fontweight="bold")

# ---------------------------------------------------------------- panel b data
lad = json.load(open(os.path.join(W, "cell_ladder_result.json")))["grid"]
order_obs = ["single", "fi7", "trace1", "trace5"]
obs_label = ["1", "7", "200", "1000"]
npars = [3, 5, 8]
err = np.full((len(npars), len(order_obs)), np.nan)
eff = np.full_like(err, np.nan)
matched = np.zeros_like(err, dtype=bool)
for i, p in enumerate(npars):
    for j, o in enumerate(order_obs):
        cell = lad.get(f"{o}_p{p}", {})
        r = cell.get("recovery", {})
        if "rel_theta_median" in r:
            err[i, j] = r["rel_theta_median"]
            matched[i, j] = r.get("L_final_median", 1.0) < 0.02
        c = cell.get("curvature", {})
        if "eff_dim_90" in c:
            eff[i, j] = c["eff_dim_90"]

b = ax[1]
im = b.imshow(err, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
    "bw", ["#FFFFFF", BLUE]), vmin=0, vmax=0.6, aspect="auto")
for i in range(len(npars)):
    for j in range(len(order_obs)):
        if np.isfinite(err[i, j]):
            dark = err[i, j] > 0.33
            star = "" if matched[i, j] else "†"
            b.text(j, i - 0.13, f"{err[i,j]*100:.0f}%{star}", ha="center",
                   va="center", fontsize=6.4, color="white" if dark else INK)
            b.text(j, i + 0.24, f"dim {eff[i,j]:.0f}", ha="center", va="center",
                   fontsize=5.0, color="white" if dark else GREY)
b.set_xticks(range(len(order_obs)))
b.set_xticklabels(obs_label)
b.set_yticks(range(len(npars)))
b.set_yticklabels([str(p) for p in npars])
b.set_xlabel("Number of observables")
b.set_ylabel("Number of parameters")
for s in b.spines.values():
    s.set_visible(False)
b.tick_params(length=0)
cb = fig.colorbar(im, ax=b, fraction=0.045, pad=0.03)
cb.set_label("Recovery error", fontsize=6)
cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
cb.outline.set_linewidth(0.5)
b.text(-0.20, 1.02, "c", transform=b.transAxes, fontsize=8, fontweight="bold")
b.text(0.0, -0.32, "† behaviour never matched: the error reflects the optimiser, "
       "not identifiability", transform=b.transAxes, fontsize=5.2, color=GREY)

p = os.path.join(FIG, "fig_p2_cell_groundtruth.png")
fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close(fig)
from PIL import Image
w, h = Image.open(p).size
print(f"fig_p2_cell_groundtruth.png  {w}x{h}  = {w/300*25.4:.0f} mm at 300 dpi")
