"""Random-subspace null for the displaced-operating-point comparison (S5.3).

The observed mean pairwise principal cosine between stiff subspaces at the five
operating points is 0.89. This computes the chance level: the mean principal
cosine between independently drawn random orthonormal frames of the measured
dimensions (8 or 9) in the 300-dimensional probed space.

    python scripts/_null_subspace_overlap.py     writes results/null_subspace_overlap.json
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "null_subspace_overlap.json")

N = 300                     # probed connections per operating point (E58)
DIMS = [9, 10, 8, 12, 8]    # measured 90% dimensions at the five points
NDRAW = 4000
rng = np.random.default_rng(20260828)

vals = []
pairs = [(a, b) for i, a in enumerate(DIMS) for b in DIMS[i + 1:]]
for _ in range(NDRAW // len(pairs)):
    for ka, kb in pairs:
        A = np.linalg.qr(rng.standard_normal((N, ka)))[0]
        B = np.linalg.qr(rng.standard_normal((N, kb)))[0]
        s = np.linalg.svd(A.T @ B, compute_uv=False)
        vals.append(float(s.mean()))
vals = np.asarray(vals)

out = {"experiment": "null_subspace_overlap",
       "question": "chance level for the mean pairwise principal cosine of E58's stiff subspaces",
       "ambient_dim": N, "subspace_dims": DIMS, "n_draws": len(vals),
       "seed": 20260828,
       "mean": round(float(vals.mean()), 4), "sd": round(float(vals.std()), 4),
       "p95": round(float(np.percentile(vals, 95)), 4)}
json.dump(out, open(OUT, "w"), indent=2)
print("null mean %.3f +/- %.3f over %d draws" % (out["mean"], out["sd"], out["n_draws"]))
