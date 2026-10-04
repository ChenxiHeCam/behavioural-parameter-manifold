#!/usr/bin/env python3
"""One-command reproduction of every quantitative claim and figure in the
accompanying manuscript, from the deposited result JSONs. Run: python regenerate_all.py

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
# the modWorm participations are computed from eigenvectors at 1e-15, so the paper
# takes this measurement on BAAIWorm instead; both are checked, and the modWorm
# spectrum is checked to be the degenerate one that motivates the switch
check("modWorm trailing eigenvalues are numerically zero",
      max(abs(v) for v in cp["hessian_eigvals"][-2:]) < 1e-14, True, 0.01)
_ab7 = L("AB7_baai_participation.json")
check("AB7 all 16 BAAIWorm eigenvectors resolved", _ab7["n_resolved_eigenvectors"], 16, 0.01)
check("AB7 stiffest participation = 2.88", _ab7["stiffest"]["participation"], 2.88, 0.02)
check("AB7 sloppiest participation = 1.02", _ab7["sloppiest_resolved"]["participation"], 1.02, 0.02)
check("AB7 sloppiest eigenvalue is five orders above the floor",
      _ab7["sloppiest_resolved"]["eigenvalue"] / _ab7["resolution_floor"] > 1e4, True, 0.01)

# ---- D1 multipoint, D3 stats ----
d1 = L("mw_D1R6.json")["D1_multipoint_hessian"]["per_point"]
d1seq = [d1[f"point_{i}"]["eff_dim_99"] for i in range(5)]
check("D1 multipoint eff99 = [4,3,1,3,4] sum=15", sum(d1seq), 15, 0.01)
d3 = L("D3_modworm_stats.json")
check("D3 Spearman rho = 0.964", d3["spearman_rho"], 0.9643, 0.001)
check("D3 exact-perm p = 0.0028", d3["p_exact_permutation"], 0.002778, 0.0005)

# ---- threshold-robustness table (S6.9) ----
# ---- per-channel decomposition (S5.2): every named mechanism must respond ----
e45 = L("E45b_baai_named_channels.json")
_el = [abs(r[1]) for r in e45["ranking_stiff_to_sloppy"]]
check("S5.2 named mechanisms = 16", e45["n_mechanisms"], 16, 0.01)
check("S5.2 no mechanism identically zero", min(_el) > 0 and not e45["identically_zero"], True, 0.01)
check("S5.2 eff-dim 90% = 3", e45["eff_dim_90"], 3, 0.01)
check("S5.2 eff-dim 99% = 6", e45["eff_dim_99"], 6, 0.01)

# ---- pyloric: the withdrawal must stay withdrawn (S2.1) ----
ab2 = L("AB2_stg_scan.json")
_c = ab2["coarse_15_summary_stats"]
check("S2.1 K-scan usable points = 24", _c["n"], 24, 0.01)
check("S2.1 K-scan mean eff-dim = 2.33", _c["mean"], 2.333, 0.01)
check("S2.1 the retired 2.13 sits within one sd of it (sample size was not the problem)",
      abs(2.13 - _c["mean"]) <= _c["sd"], True, 0.01)

# the trace response must be flat in the step: that is what makes J scale as 1/delta
_tr = {r["delta"]: r["mean"] for r in L("AB2b_stg_scaling.json")["trace_response"]}
_lo, _hi = _tr[min(_tr)], _tr[max(_tr)]
check("S1.5 pyloric trace response flat across a 250-fold step range",
      max(_lo, _hi) / min(_lo, _hi) < 1.3, True, 0.01)

# and the summary statistics must fail proportionality too, under both whitenings:
# that is what retires the pyloric dimension, not the sample size
_w = L("AB2d_stg_whitening.json")
_sat = _w["saturated_reference_spread"]
for _tag in ("baseline", "population"):
    _r = _w["ratio_spread_" + _tag]
    check("S2.1 summary stats saturated under " + _tag + " whitening",
          _r["median"] > 0.8 * _sat, True, 0.01)
    check("S2.1 few pairs proportional under " + _tag + " whitening",
          _r["fraction_proportional"] < 0.10, True, 0.01)

# the population geometry that replaces it needs no finite difference at all
# ---- external datasets that use none of our forward models (S2.2-S2.4) ----
_al = L("E2_allen_variability.json")
check("S2.2 Allen cohort is 1914 complete cases", _al["n_cells"], 1914, 0.5)
check("S2.2 Allen property participation dimension 4.78 of 11",
      _al["participation_eff_dim"], 4.78, 0.02)
_cvr = [v for k, v in _al["cv_per_feature"].items()
        if k not in ("ef__vrest", "ef__fast_trough_v_long_square")]
check("S2.2 ratio-scale CV runs 0.34 to 1.85 (voltages excluded)",
      round(min(_cvr), 2), 0.34, 0.01)
check("S2.2 the most variable property is adaptation at 1.85",
      round(max(_cvr), 2), 1.85, 0.01)

_gn = L("E4_channel_conservation.json")["genes"]
check("S2.3 forty worm channel and synaptic genes", len(_gn), 40, 0.5)
check("S2.3 unc-9 is the most conserved at 99.91",
      max(g["mean_perc_id"] for g in _gn), 99.91, 0.02)
check("S2.3 unc-49 is the least conserved at 77.94",
      min(g["mean_perc_id"] for g in _gn), 77.94, 0.02)

_n2 = L("E5_n2_manifold.json")
check("S2.4 sixty-one isogenic N2 animals", _n2["n_worms"], 61, 0.5)
check("S2.4 cross-individual participation dimension 9.75",
      _n2["participation_eff_dim"], 9.75, 0.02)
check("S2.4 the feature count exceeds the animal count, so rank is capped",
      _n2["n_features"] > _n2["n_worms"], True, 0.01)

_em = L("E1_pyloric_manifold.json")
check("S2.1 pyloric population participation dim = 23.2", _em["participation_eff_dim"], 23.157, 0.01)
check("S2.1 pyloric population 90% dim = 23", _em["eff_dim_90pct"], 23, 0.01)

# ---- FlyGym: reported at a step where the response is proportional (S5.1) ----
_fg = L("E1_flygym_rank.json")
check("S5.1 FlyGym coarse eff-dim 90% = 1", _fg["coarse"]["eff_dim_90"], 1, 0.01)
_ab4 = L("AB4_flygym_delta.json")
_by = {r["delta"]: r for r in _ab4["by_delta"]}
check("S5.1 FlyGym rich eff-dim 90% = 11 at the valid step",
      _by[0.25]["rich"]["eff_dim_90"], 11, 0.01)
check("S5.1 FlyGym rich eff-dim 99% = 31 at the valid step",
      _by[0.25]["rich"]["eff_dim_99"], 31, 0.01)
check("S5.1 the 0.05 step inflates it to 16", _by[0.05]["rich"]["eff_dim_90"], 16, 0.01)
check("S1.5 FlyGym proportional across the valid region",
      _ab4["ratio_spread_valid_region"] < 1.10, True, 0.01)
check("S1.5 FlyGym not proportional across the full range",
      _ab4["ratio_spread_all_steps"] > 1.40, True, 0.01)
check("S5.1 FlyGym coarse reading is 1/1 at every step",
      all(r["coarse"]["eff_dim_90"] == 1 for r in _ab4["by_delta"]), True, 0.01)
# the saturation curve at the same valid step
_e36 = L("E36b_flygym_validstep.json")["curve"]
check("S6.5 FlyGym union 12.25 at one behaviour", _e36[0]["eff_dim_90_mean"], 12.25, 0.01)
check("S6.5 FlyGym union 13.00 at four behaviours", _e36[-1]["eff_dim_90_mean"], 13.0, 0.01)
check("S6.5 the increment is under half the single-behaviour spread",
      (_e36[-1]["eff_dim_90_mean"] - _e36[0]["eff_dim_90_mean"]) < 0.5 * _e36[0]["eff_dim_90_sd"],
      True, 0.01)

# ---- the class-gain probe reaches little of the per-cell curvature (S5.2.1) ----
check("S5.2.1 uniform direction carries 0.295% of the curvature",
      L("E37_granularity_modworm.json")["fraction_of_curvature_in_uniform_direction"],
      0.002953, 0.01)

# ---- union: rank grows, effective dimension does not (S6.10) ----
_b6 = L("B6_union_crossbehaviour.json")
check("S6.10 union eff-dim equals chemotaxis alone",
      _b6["eff_dim_UNION_90_99"] == _b6["eff_dim_chemotaxis_90_99"], True, 0.01)
_eig = _b6["eig_union"]
check("S6.10 the two extra rank directions carry 0.56% of the mass",
      (_eig[4] + _eig[5]) / sum(_eig), 0.00557, 0.02)
check("S6.10 the fifth direction carries 0.30% of the mass",
      100 * _eig[4] / sum(_eig), 0.30, 0.02)
check("S6.10 the sixth direction carries 0.26% of the mass",
      100 * _eig[5] / sum(_eig), 0.26, 0.02)
check("S6.10 the third direction carries 1.51% of the mass",
      100 * _eig[2] / sum(_eig), 1.51, 0.02)
check("S6.10 the fourth direction carries 1.03% of the mass",
      100 * _eig[3] / sum(_eig), 1.03, 0.02)
check("S6.10 the leading union direction carries 87% on its own",
      100 * _eig[0] / sum(_eig), 87.1, 0.05)

# ---- sloppy directions move behaviour more, not less, at a finite step (S6.7) ----
_cp = L("CP_coupling.json")["behaviour_change_under_moves"]
check("S6.7 sloppy eigenvector moves behaviour more than stiff at alpha=0.6",
      _cp["sloppy_eigvec_alpha0.6"] > _cp["stiff_eigvec_alpha0.6"], True, 0.01)

# ---- Holm correction over the S6.4 family leaves both survivors significant ----
_ps = sorted([0.0157, 0.0204, 0.08185])
_holm = [pv * (len(_ps) - i) for i, pv in enumerate(_ps)]
check("S6.4 Holm-corrected cosine p = 0.047", _holm[0], 0.0471, 0.01)
check("S6.4 Holm-corrected span p = 0.041", _holm[1], 0.0408, 0.01)
check("S6.4 Holm-corrected novel fraction stays non-significant", _holm[2] > 0.05, True, 0.01)

print("\n--- S6.9 threshold robustness (recomputed) ---")
spectra = {
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
