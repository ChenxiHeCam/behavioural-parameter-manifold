#!/usr/bin/env python3
"""One-command reproduction of every quantitative claim and figure in Paper 2
from the deposited result JSONs. Run: python regenerate_all.py

It (1) recomputes effective dimension / participation from the saved eigenspectra,
(2) machine-checks each number cited in the main text and SI against its JSON,
(3) regenerates the cross-species/tiling hero figure, and
(4) prints a PASS/FAIL audit. No simulator or server is required: the forward-model
runs are deposited as result files; this script reproduces the analysis on top of them.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RESULTS_DIR", os.path.join(HERE, "..", "results"))  # JSONs live one level up

def L(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)

def effdim(eigs, thr):
    a = np.sort(np.abs(np.asarray(eigs, float)))[::-1]; a = a[a > 0]
    if a.sum() == 0: return 0
    return int(np.searchsorted(np.cumsum(a) / a.sum(), thr) + 1)

def partic(eigs):
    a = np.abs(np.asarray(eigs, float)); a = a[a > 0]
    return float(a.sum() ** 2 / (a ** 2).sum())

checks = []
def check(name, got, want, tol=0.02):
    ok = abs(float(got) - float(want)) <= tol
    checks.append((name, ok, got, want))

# ---- eigenworm identity ----
ew = L("EW_eigenworm.json"); ewb = L("EW_baai_local.json")
check("real posture eff-dim (49pt) = 3.79", ew["real_posture_effdim"], 3.79, 0.02)
check("real posture eff-dim (17pt) = 4.14", ewb["real_posture_effdim"], 4.14, 0.02)
check("4 eigenworms cum-var (49pt) = 0.96", ew["real_varexp_top4"][3], 0.96, 0.01)
check("modWorm eigenworm align cos = 0.77", ew["eigenworm_subspace_alignment"]["mean_cos_principal_angles_top4"], 0.773, 0.01)
check("BAAIWorm-sim align cos = 0.68", ewb["baai_arms"]["sim"]["subspace_align_to_real_mean_cos"], 0.683, 0.01)
check("BAAIWorm-sim var in real top4 = 0.72", ewb["baai_arms"]["sim"]["frac_baai_variance_in_real_top4_eigenworms"], 0.719, 0.01)

# ---- complementary tiling ----
ch = L("BAAI_chemo.json")
check("chemo context ratio = 7.15", ch["chemo_food_vs_nofood_ratio"], 7.15, 0.05)
check("motor context ratio = 0.30", ch["motor_food_vs_nofood_ratio"], 0.305, 0.02)

# ---- saturation / multibehaviour ----
sat = L("SAT_saturation.json")
check("modWorm 12-behaviour union eff99 = 4", sat["saturation_curve"][-1]["eff_dim_99_mean"], 4.0, 0.01)
check("modWorm 12-behaviour union eff90 = 2", sat["saturation_curve"][-1]["eff_dim_90_mean"], 2.0, 0.01)
mb = L("MB_behaviour_specificity.json")
check("4-preset union eff99 = 5", mb["union_eff_dim_90_99"][1], 5, 0.01)
check("cross-behaviour mean cos = 0.54", mb["mean_cross_behaviour_alignment"], 0.5395, 0.01)

# ---- cross-species ----
la = L("LARVA_manifold.json"); ro = L("RODENT_manifold.json")
check("larva union eff99 = 8", la["union_eff_dim_90_99"][1], 8, 0.01)
check("rodent union eff99 = 12", ro["union_eff_dim_90_99"][1], 12, 0.01)
check("rodent 6-behaviour curve end = 12", ro["saturation_curve"][-1]["eff_dim_99_mean"], 12.0, 0.01)

# ---- G2 two-model agreement ----
g2 = L("G2_crosssim_align.json")["PRIMARY_metric_wiring_vs_singlecell"]
check("modWorm wiring stiff-pct = 0.833", g2["modWorm"]["wiring"], 0.833, 0.01)
check("BAAIWorm wiring stiff-pct = 0.868", g2["BAAIWorm"]["wiring"], 0.868, 0.01)
check("modWorm single-cell stiff-pct = 0.083", g2["modWorm"]["single_cell"], 0.083, 0.01)

# ---- CP coupling ----
cp = L("CP_coupling.json")
check("stiff eigvec participation = 3.42", cp["stiff_eigvec_participation"], 3.416, 0.02)
check("sloppy eigvec participation = 4.68", cp["sloppy_eigvec_participation"], 4.682, 0.02)

# ---- D1 multipoint, D3 stats ----
d1 = L("mw_D1R6.json")["D1_multipoint_hessian"]["per_point"]
d1seq = [d1[f"point_{i}"]["eff_dim_99"] for i in range(5)]
check("D1 multipoint eff99 = [4,3,1,3,4] sum=15", sum(d1seq), 15, 0.01)
d3 = L("D3_modworm_stats.json")
check("D3 Spearman rho = 0.964", d3["spearman_rho"], 0.9643, 0.001)
check("D3 exact-perm p = 0.0028", d3["p_exact_permutation"], 0.002778, 0.0005)

# ---- threshold-robustness table (S6.9) ----
print("\n--- S6.9 threshold robustness (recomputed) ---")
spectra = {
    "modWorm full":   L("modworm_hessian_full.json")["eig"],
    "modWorm 7-mech": cp["hessian_eigvals"],
    "rich-observable": L("RICH_observable.json")["results_by_duration"]["NSTEP300"]["spectrum_normalised"],
    "larva union":    la["union_spectrum"],
    "rodent union":   ro["union_spectrum_top10"],
}
print(f"{'system':18s} {'80/90/95/99':>14s} {'PR':>6s}")
for k, e in spectra.items():
    ed = "/".join(str(effdim(e, t)) for t in (.80, .90, .95, .99))
    print(f"{k:18s} {ed:>14s} {partic(e):6.2f}")

# ---- regenerate hero figure ----
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
    ax[0].plot(range(1, 5), ew["real_varexp_top4"], "o-"); ax[0].set_title("a eigenworm")
    c = ch["conditions"]; x = np.arange(2); w = .36
    ax[1].bar(x - w/2, [c["close"]["chemo_perturb_behav_change"], c["far"]["chemo_perturb_behav_change"]], w, label="chemo")
    ax[1].bar(x + w/2, [c["close"]["motor_perturb_behav_change"], c["far"]["motor_perturb_behav_change"]], w, label="motor")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["chemotaxis", "locomotion"]); ax[1].legend(); ax[1].set_title("b tiling")
    nb = [r["n_behaviours"] for r in sat["saturation_curve"]]; e99 = [r["eff_dim_99_mean"] for r in sat["saturation_curve"]]
    ax[2].plot(nb, e99, "o-"); ax[2].set_title("c saturation (worm)")
    rc = ro["saturation_curve"]; ax[3].plot([r["n_behaviours"] for r in rc], [r["eff_dim_99_mean"] for r in rc], "s-", label="rodent")
    ax[3].plot(nb, e99, "o-", label="worm"); ax[3].legend(); ax[3].set_title("d union across phyla")
    plt.tight_layout(); out = os.path.join(HERE, "fig_p2_tiling_crossspecies_CHECK.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"\nregenerated check figure {os.path.basename(out)} (publication-quality version is produced by scripts/_fig_nc_tiling.py)")
except Exception as e:
    print(f"\n[figure skipped: {e}]")

# ---- audit summary ----
npass = sum(ok for _, ok, _, _ in checks)
print(f"\n=== NUMBER AUDIT: {npass}/{len(checks)} PASS ===")
for name, ok, got, want in checks:
    if not ok:
        print(f"  FAIL  {name}: got {got} want {want}")
sys.exit(0 if npass == len(checks) else 1)
