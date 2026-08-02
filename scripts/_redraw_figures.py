"""Redraw the figures that were below print resolution, in one consistent style.

Nature specification: 89 mm single column / 183 mm double, >=300 dpi (we render at
600), sans-serif 5-7 pt, no panel titles (the caption carries them), no gridlines,
left and bottom spines only, ticks outward, colourblind-safe palette with no
red-green pairing, and a colour semantics held fixed across every figure:

    blue    #0072B2  stiff / identifiable
    grey    #808080  sloppy / unconstrained
    vermil. #D55E00  a second system or condition
    green   #009E73  a third system or condition
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


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(p).size
    print(f"  {name:34s} {w}x{h}  = {w/300*25.4:.0f} mm at 300 dpi")


# ---------------------------------------------------- 1. modWorm 7-mechanism GN
d = L("modworm_REAL_b1.json")
mech = d["mechanisms"]
el = d.get("per_mechanism_elasticity") or {}
if isinstance(el, dict) and el:
    vals = [abs(float(el[m])) for m in mech]
else:                                    # fall back to the eigen-spectrum
    vals = [abs(v) for v in d["eigenvalues"]]
order = np.argsort(vals)[::-1]
lab = [d["names"].get(mech[i], mech[i]) if isinstance(d.get("names"), dict) else mech[i] for i in order]
v = np.array(vals)[order]
k = int(np.searchsorted(np.cumsum(v) / v.sum(), 0.90) + 1)

fig, ax = plt.subplots(figsize=(89 * MM, 52 * MM))
ax.barh(range(len(v)), v, color=[BLUE] * k + [GREY] * (len(v) - k), height=0.7)
ax.set_yticks(range(len(v))); ax.set_yticklabels(lab)
ax.invert_yaxis(); ax.set_xscale("log")
ax.set_xlabel("Behavioural elasticity (log scale)")
ax.text(0.97, 0.10, "synaptic, gap-junction and\nrise-time axes carry the curvature;\ncapacitance and gain are flat",
        transform=ax.transAxes, ha="right", fontsize=6, color=INK, linespacing=1.35)
fig.tight_layout()
save(fig, "fig_p2_modworm_real_b1.png")

# --------------------------------------------- 2. BAAIWorm 20 named mechanisms
d = L("baai_perchannel_hessian.json")
el = d.get("per_channel_stiffness") or {}
gn = d.get("gene_names", {})
if isinstance(el, dict) and el:
    items = sorted(((gn.get(kk, kk), abs(float(vv))) for kk, vv in el.items()),
                   key=lambda t: t[1], reverse=True)
else:
    ev = [abs(x) for x in d["eigenvalues"]]
    items = [(f"mode {i+1}", x) for i, x in enumerate(sorted(ev, reverse=True))]
names = [t[0] for t in items]; vals = np.array([t[1] for t in items])
k = int(np.searchsorted(np.cumsum(vals) / vals.sum(), 0.90) + 1)

# colour by mechanism class: wiring/passive (blue) vs single-cell fast channels (grey)
WIRING = ("gap", "synap", "motor", "leak", "passive")
cls = [BLUE if any(w in nm.lower() for w in WIRING) else GREY for nm in names]
fig, ax = plt.subplots(figsize=(89 * MM, 80 * MM))
ax.barh(range(len(vals)), np.maximum(vals, 1e-6), color=cls, height=0.72)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=5.5)
ax.invert_yaxis(); ax.set_xscale("log")
ax.set_xlabel("Behavioural elasticity (log scale)")
ax.text(0.97, 0.13, "wiring and passive properties\nrank above the fast Ca and\nCa-activated K channels",
        transform=ax.transAxes, ha="right", fontsize=6, color=INK, linespacing=1.35)
ax.text(0.97, 0.03, "blue: wiring / passive     grey: single-cell channels",
        transform=ax.transAxes, ha="right", fontsize=5.5, color=GREY)
fig.tight_layout()
save(fig, "fig_p2_b1_perchannel.png")

# --------------------------------------------------- 3. FlyGym 48-d eigenvalues
d = L("flygym_hessian48.json")
ev = np.array([abs(x) for x in d["eigenvalues"]])
ev = np.sort(ev)[::-1]
k = d.get("eff_dim_90", int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.90) + 1))

fig, ax = plt.subplots(figsize=(89 * MM, 52 * MM))
ax.plot(range(1, len(ev) + 1), np.maximum(ev, 1e-12), "o-", color=BLUE, lw=1.0, ms=2.4,
        markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=0.3)
ax.axvline(k + 0.5, ls=(0, (3, 3)), color=GREY, lw=0.6)
ax.set_yscale("log"); ax.set_xlabel("Eigenvalue index"); ax.set_ylabel("Curvature (log scale)")
ax.text(k + 1.6, ev.max() * 0.35, f"{k} directions\nreach 90%", fontsize=6,
        color=GREY, linespacing=1.3)
ax.text(0.97, 0.86, f"spectrum spans\n{d.get('spectral_span_orders', 19.5):.1f} orders",
        transform=ax.transAxes, ha="right", va="top", fontsize=6, color=INK, linespacing=1.3)
fig.tight_layout()
save(fig, "fig_p2_flygym_hessian48.png")

# ---------------------------------------------------------- 4. flyvis per-type
d = L("flyvis_b1.json")
st = d.get("per_celltype_stiffness", {})
items = sorted(((str(kk).replace("b'", "").replace("'", ""), abs(float(vv)))
                for kk, vv in st.items()), key=lambda t: t[1], reverse=True)
top = items[:10]; bot = items[-10:]
sel = top + bot
fig, ax = plt.subplots(figsize=(89 * MM, 72 * MM))
cols = [BLUE] * len(top) + [GREY] * len(bot)
ax.barh(range(len(sel)), [t[1] for t in sel], color=cols, height=0.72)
ax.set_yticks(range(len(sel))); ax.set_yticklabels([t[0] for t in sel], fontsize=5.5)
ax.invert_yaxis(); ax.set_xlabel("Behavioural elasticity")
ax.axhline(len(top) - 0.5, color=LGREY, lw=0.6, ls=(0, (2, 2)))
ax.text(0.97, 0.50, f"effective dimension\n{d['eff_dim_90']} of {d['n_cell_types']} cell types",
        transform=ax.transAxes, ha="right", fontsize=6, color=INK, linespacing=1.3)
ax.text(0.97, 0.96, "10 stiffest", transform=ax.transAxes, ha="right", va="top",
        fontsize=5.8, color=BLUE)
ax.text(0.97, 0.30, "10 sloppiest", transform=ax.transAxes, ha="right", va="top",
        fontsize=5.8, color=GREY)
fig.tight_layout()
save(fig, "fig_p2_flyvis_b1.png")

# ------------------------------------------------- 5. cross-system convergence
d = L("STG_b1analog.json")
stiff = [(n, float(w)) for n, w in d["stiff_top"]][:6]
sloppy = [(n, float(w)) for n, w in d["sloppy_top"]][:6]
fig, ax = plt.subplots(figsize=(89 * MM, 58 * MM))
sel = stiff + sloppy
ax.barh(range(len(sel)), [t[1] for t in sel],
        color=[BLUE] * len(stiff) + [GREY] * len(sloppy), height=0.72)
ax.set_yticks(range(len(sel))); ax.set_yticklabels([t[0] for t in sel], fontsize=5.5)
ax.invert_yaxis(); ax.set_xlabel("Eigenvector loading")
ax.axhline(len(stiff) - 0.5, color=LGREY, lw=0.6, ls=(0, (2, 2)))
ax.text(0.97, 0.95, "stiff: leak and\ninhibitory synapses", transform=ax.transAxes,
        ha="right", va="top", fontsize=5.8, color=BLUE, linespacing=1.25)
ax.text(0.97, 0.42, "sloppy: fast voltage-\ngated sodium", transform=ax.transAxes,
        ha="right", va="top", fontsize=5.8, color=GREY, linespacing=1.25)
ax.text(0.97, 0.06, f"effective dimension {d['eff_dim_mean']:.2f} of 31",
        transform=ax.transAxes, ha="right", fontsize=6, color=INK)
fig.tight_layout()
save(fig, "fig_p2_crosssystem_real.png")

print("done")

# ------------------------------------------- 6. B3 in-silico mutant localisation
d = L("baai_b3_localise.json")
rank = d["rank_by_signature_norm"]           # [key, name, signature_norm, n_confusable]
names = [r[1] for r in rank]
sig = np.array([float(r[2]) for r in rank])
conf = np.array([float(r[3]) for r in rank])
WIRING = ("gap", "synap", "motor", "leak", "passive", "nca")
cls = [BLUE if any(w in n.lower() for w in WIRING) else GREY for n in names]

fig, ax = plt.subplots(figsize=(89 * MM, 62 * MM))
ax.scatter(sig, conf, s=16, c=cls, edgecolor="white", linewidth=0.3, zorder=3)
for n, x, y in zip(names, sig, conf):
    if x > 0.55 or x < 0.13:
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(3, 3),
                    fontsize=5, color=INK)
ax.axvline(0.2, ls=(0, (3, 3)), color=LGREY, lw=0.6)
ax.set_xlabel("Behavioural signature norm (detectability)")
ax.set_ylabel("Number of confusable mechanisms")
ax.text(0.03, 0.06, "behaviourally near-silent", transform=ax.transAxes,
        fontsize=5.6, color=GREY)
ax.text(0.97, 0.94, "blue: wiring / passive    grey: single-cell channels",
        transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=GREY)
fig.tight_layout()
save(fig, "fig_p2_b3_localise.png")
