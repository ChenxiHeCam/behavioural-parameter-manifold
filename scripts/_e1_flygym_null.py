"""Noise control for E-1.

Removing the observable rank cap raised the FlyGym effective dimension from ~1 to
16/38 of 48. Before that can be read as signal, it has to be separated from
finite-difference noise: with 4200 observables and a stochastic controller, seed
jitter alone can populate many directions.

Here the same estimator is run on a Jacobian built entirely from seed-to-seed
variation at the *unperturbed* parameter value. Any effective dimension it
produces is noise. If the null is comparable to 16/38, the rise is an artefact;
if it is small, the rise is real.
"""
import json, os, sys, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/root/autodl-tmp")
from e1_flygym import rollout, eff_dim, DELTA, NSEED

NSEED_NULL = 20   # independent seeds at the base parameter value


def main():
    t0 = time.time()
    jobs = [(-1, 0, 1000 + s) for s in range(NSEED_NULL * 2)]
    print(f"{len(jobs)} base rollouts for the noise null", flush=True)
    outs = []
    with ProcessPoolExecutor(max_workers=min(40, os.cpu_count() - 4)) as ex:
        for n, o in enumerate(ex.map(rollout, jobs, chunksize=1), 1):
            outs.append(o[4])
            if n % 10 == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    T = min(a.shape[0] for a in outs)
    A = np.array([a[:T] for a in outs], dtype=float)

    # build 48 pseudo-columns the same way the real Jacobian is built, but from
    # disjoint seed groups at identical parameters: the "perturbation" is zero,
    # so every column is pure noise of the same magnitude as the real estimate
    rng = np.random.default_rng(0)
    cols = []
    for i in range(48):
        ip = rng.choice(NSEED_NULL, NSEED, replace=False)
        im = rng.choice(np.arange(NSEED_NULL, 2 * NSEED_NULL), NSEED, replace=False)
        jp = A[ip].mean(0)
        jm = A[im].mean(0)
        cols.append(((jp - jm) / (2 * DELTA)).flatten())
    J = np.array(cols).T

    base = A.mean(0)
    sj = base.std(0).clip(1e-6)
    Jw = J / np.tile(sj, T)[:, None]
    H = Jw.T @ Jw
    n90, n99, npr, nspec = eff_dim(np.linalg.eigvalsh(H))

    out = {
        "experiment": "E1_noise_null",
        "question": "how much effective dimension does seed noise alone produce "
                    "in the rich-observable estimator?",
        "n_seeds_per_group": NSEED, "n_independent_seeds": NSEED_NULL * 2,
        "n_observables": int(T * 42),
        "null_eff_dim_90": n90, "null_eff_dim_99": n99,
        "null_participation_ratio": npr, "null_top_eigenvalues": nspec,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open("/root/autodl-tmp/E1_flygym_null.json", "w"), indent=2)
    print("\n=== E-1 NOISE NULL ===", flush=True)
    print(f"noise-only  -> eff-dim {n90}/{n99} of 48   PR {npr:.2f}", flush=True)
    print("NULL_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
