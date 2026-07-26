"""EW (BAAIWorm arm, fully local): does BAAIWorm posture live in the same low-dim
eigenworm space as real OWMD N2? Both in 17-keypoint format (BAAIWorm_pgob npz + owmd_real_tensor).
Real eigenworms from 5010 windows (robust); BAAIWorm 100-frame posture projected onto them.
No simulator re-run, no server."""
import numpy as np, json
from scipy.linalg import subspace_angles

KEW = 4


def angles_from_kp(X):  # X (...,P,2) -> tangent-angle profile (...,P-1), orientation-removed
    d = np.diff(X, axis=-2)
    a = np.arctan2(d[..., 1], d[..., 0])
    a = np.unwrap(a, axis=-1)
    return a - a.mean(-1, keepdims=True)


def eigenworms(A):
    A = A - A.mean(0, keepdims=True)
    w, V = np.linalg.eigh(np.cov(A.T))
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def eff_dim(w):
    w = w[w > 0]
    return float(w.sum() ** 2 / (w ** 2).sum())


# real OWMD N2 (17-keypoint tensor)
RT = np.load('owmd_real_tensor.npz')['X'][..., :2]      # (5010,100,17,2)
real_kp = RT.reshape(-1, 17, 2)
# subsample for speed + drop non-finite
real_kp = real_kp[np.all(np.isfinite(real_kp), axis=(1, 2))]
rng = np.random.default_rng(0)
sel = rng.choice(len(real_kp), size=min(40000, len(real_kp)), replace=False)
real_ang = angles_from_kp(real_kp[sel])                  # (N,16)
w_r, V_r = eigenworms(real_ang)
ve_r = np.cumsum(w_r) / w_r.sum()
effdim_r = eff_dim(w_r)

# BAAIWorm posture (both sim and real arms)
results = {}
for arm in ['sim', 'real']:
    b = np.load(f'BAAIWorm_pgob_{arm}_GENUINE.npz', allow_pickle=True)['X'][0, :, :, :2]  # (100,17,2)
    b = b[np.all(np.isfinite(b), axis=(1, 2))]
    ba_ang = angles_from_kp(b)                            # (100,16)
    w_b, V_b = eigenworms(ba_ang)
    effdim_b = eff_dim(w_b)
    # subspace alignment to real eigenworms
    pa = subspace_angles(V_b[:, :KEW], V_r[:, :KEW])
    mean_cos = float(np.cos(pa).mean())
    per_mode = [float(abs(np.dot(V_b[:, i], V_r[:, i]))) for i in range(KEW)]
    # fraction of BAAIWorm posture variance captured by REAL top-4 eigenworms
    proj = (ba_ang - ba_ang.mean(0)) @ V_r[:, :KEW]
    frac_in_real4 = float(np.var(proj, axis=0).sum() / np.var(ba_ang, axis=0).sum())
    results[arm] = {'baai_posture_effdim': effdim_b,
                    'subspace_align_to_real_mean_cos': mean_cos,
                    'per_mode_abs_cos': per_mode,
                    'frac_baai_variance_in_real_top4_eigenworms': frac_in_real4}

out = {
    'experiment': 'EW BAAIWorm arm (local): BAAIWorm posture vs real OWMD N2 eigenworms (17-keypoint)',
    'real_source': 'owmd_real_tensor.npz (5010 windows, 40k frames sampled)',
    'real_posture_effdim': effdim_r,
    'real_varexp_top4': [float(v) for v in ve_r[:4]],
    'baai_arms': results,
    'note': 'eigenworms from real worm posture; BAAIWorm posture projected onto the real eigenworm modes',
}
json.dump(out, open('NEXT_PAPER_manifold_subspace/EW_baai_local.json', 'w'), indent=2)
print('=== EW BAAIWorm (local) ===')
print('REAL OWMD N2 posture eff-dim %.2f | top-4 var explained %s' % (effdim_r, [round(v, 3) for v in ve_r[:4]]))
for arm, r in results.items():
    print(f'BAAIWorm-{arm}: eff-dim {r["baai_posture_effdim"]:.2f} | align-to-real cos {r["subspace_align_to_real_mean_cos"]:.3f} '
          f'| {r["frac_baai_variance_in_real_top4_eigenworms"]*100:.0f}% variance in real top-4 eigenworms')
