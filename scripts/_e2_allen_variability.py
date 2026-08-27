"""E2: stiff/sloppy structure in real cortical neurons (Allen Cell Types).

Reads the Allen Cell Types electrophysiology table and reports, for the eleven
intrinsic properties the database defines for every cell, the per-property
coefficient of variation and the intrinsic dimension of the property space.

Cells enter only if all eleven properties are present, so no imputation is used.
The coefficient of variation is computed on the raw values, since it is a
scale-free measure and the comparison is between properties. The dimension is
computed on z-scored properties, because the eleven carry different units and an
unstandardised covariance would be dominated by whichever has the largest
numerical range.

    python scripts/_e2_allen_variability.py <path to allen_celltypes_detail.json>
"""
import json, os, sys
import numpy as np

FEATURES = ["ef__adaptation", "ef__avg_firing_rate", "ef__avg_isi",
            "ef__f_i_curve_slope", "ef__fast_trough_v_long_square",
            "ef__peak_t_ramp", "ef__ri", "ef__tau",
            "ef__threshold_i_long_square",
            "ef__upstroke_downstroke_ratio_long_square", "ef__vrest"]

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "allen_celltypes_detail.json")
cells = json.load(open(src, encoding="utf-8"))

rows = []
for c in cells:
    v = [c.get(f) for f in FEATURES]
    if all(isinstance(x, (int, float)) and np.isfinite(x) for x in v):
        rows.append(v)
X = np.asarray(rows, float)
n = X.shape[0]

cv = {f: float(X[:, i].std(ddof=0) / (abs(X[:, i].mean()) + 1e-12))
      for i, f in enumerate(FEATURES)}
Z = (X - X.mean(0)) / (X.std(0, ddof=0) + 1e-12)
lam = np.linalg.svd(Z, compute_uv=False) ** 2
var = lam / lam.sum()
pr = float(lam.sum() ** 2 / (lam ** 2).sum())
cum = np.cumsum(var)

out = {
    "experiment": "E2_allen_variability",
    "source": "Allen Cell Types database, electrophysiology table "
              "(https://celltypes.brain-map.org); mouse and human cortical neurons",
    "n_cells_available": len(cells),
    "n_cells": n,
    "inclusion": "all eleven intrinsic properties present; no imputation",
    "features": FEATURES,
    "cv_per_feature": cv,
    "cv_range": [min(cv.values()), max(cv.values())],
    "cv_median": float(np.median(list(cv.values()))),
    "pca_note": "properties z-scored before PCA; the eleven carry different units",
    "pca_var_ratio": [float(x) for x in var],
    "participation_eff_dim": pr,
    "eff_dim_90pct": int(np.searchsorted(cum, 0.90) + 1),
    "eff_dim_99pct": int(np.searchsorted(cum, 0.99) + 1),
}
dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                   "E2_allen_variability.json")
json.dump(out, open(dst, "w"), indent=2)
print("n_cells %d of %d" % (n, len(cells)))
print("CV range %.3f - %.3f (median %.3f)" % (out["cv_range"][0], out["cv_range"][1], out["cv_median"]))
print("participation dimension %.3f of %d   90%%: %d   99%%: %d"
      % (pr, len(FEATURES), out["eff_dim_90pct"], out["eff_dim_99pct"]))
for f, v in sorted(cv.items(), key=lambda kv: kv[1]):
    print("   %-46s CV %.3f" % (f, v))
