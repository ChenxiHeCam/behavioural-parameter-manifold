"""E-19: what solution-cloud dimension does a locally identifiable problem give?

The paper reports that 29 of 30 FlyGym optimisations reach the behavioural target
while the solution cloud still occupies an effective 21.7 of 126 dimensions, and
reads that as evidence for a positive-dimensional equivalence manifold. But a PCA
participation dimension over N=30 points is bounded near 29, so 21.7 may simply be
what any stochastic optimiser produces, manifold or not.

The missing control is a problem with no manifold: a strictly convex quadratic in
the same dimension, fitted by the same optimiser from the same number of
independent starts. Its solution cloud is a pure measure of optimiser scatter.
Anything the real cloud has above that is attributable to degeneracy.

Reported alongside is a rank-deficient quadratic with a known 20-dimensional null
space, which calibrates what the estimator returns when the manifold dimension is
known exactly.
"""
import json, os, time
import numpy as np

DIM = 126
NRUN = 30
SIGMA0 = 0.3
MAXIT = 400
SEED = 0


def cma_es(f, dim, seed, sigma0=SIGMA0, maxit=MAXIT, popsize=None):
    """Separable CMA-ES, the optimiser family named in the paper's Methods."""
    rng = np.random.default_rng(seed)
    lam = popsize or (4 + int(3 * np.log(dim)))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / (w ** 2).sum()
    cs = (mueff + 2) / (dim + mueff + 5)
    cc = 4.0 / (dim + 4)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

    x = rng.normal(0, 1.0, dim)
    sigma = sigma0
    C = np.ones(dim)
    ps = np.zeros(dim)
    pc = np.zeros(dim)
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

    for it in range(maxit):
        z = rng.normal(0, 1, (lam, dim))
        y = z * np.sqrt(C)
        X = x + sigma * y
        fit = np.array([f(xi) for xi in X])
        idx = np.argsort(fit)
        ysel = y[idx[:mu]]
        yw = (w[:, None] * ysel).sum(0)
        x = x + sigma * yw
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * yw / np.sqrt(C)
        hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) / chiN < 1.4 + 2 / (dim + 1)
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        C = ((1 - c1 - cmu) * C + c1 * (pc ** 2)
             + cmu * (w[:, None] * ysel ** 2).sum(0))
        C = np.maximum(C, 1e-20)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        if sigma < 1e-11:
            break
    return x, float(f(x))


def participation_dim(P):
    P = P - P.mean(0)
    ev = np.linalg.svd(P, compute_uv=False) ** 2
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # A: strictly convex, unique optimum -> no manifold at all
    A = rng.normal(0, 1, (DIM, DIM))
    Hfull = A.T @ A / DIM + np.eye(DIM) * 0.5
    tgt = rng.normal(0, 1, DIM)

    def f_ident(v):
        d = v - tgt
        return float(d @ Hfull @ d)

    # B: rank-deficient, exactly 20 flat directions -> known manifold dimension
    K = 20
    Q, _ = np.linalg.qr(rng.normal(0, 1, (DIM, DIM)))
    lam = np.concatenate([np.linspace(1.0, 0.05, DIM - K), np.zeros(K)])
    Hdeg = Q @ np.diag(lam) @ Q.T

    def f_degen(v):
        d = v - tgt
        return float(d @ Hdeg @ d)

    res = {}
    for name, f in (("identifiable", f_ident), ("degenerate_k20", f_degen)):
        sols, losses = [], []
        for r in range(NRUN):
            x, fv = cma_es(f, DIM, seed=1000 + r)
            sols.append(x)
            losses.append(fv)
        P = np.array(sols)
        res[name] = {
            "n_runs": NRUN,
            "cloud_participation_dim": participation_dim(P),
            "final_loss_median": float(np.median(losses)),
            "final_loss_max": float(np.max(losses)),
            "param_dispersion": float(np.std(P, axis=0).mean()),
        }
        print(f"{name:16s} cloud dim {res[name]['cloud_participation_dim']:.1f} "
              f"of {DIM}   loss median {np.median(losses):.3e}", flush=True)

    out = {
        "experiment": "E19_dispersion_null",
        "question": "is a solution-cloud dimension of 21.7 of 126 evidence of a "
                    "manifold, or the scatter any stochastic optimiser leaves?",
        "dim": DIM, "n_runs": NRUN, "optimiser": "separable CMA-ES",
        "controls": res,
        "reported_flygym_cloud_dim": 21.7,
        "note": "the identifiable control has a unique optimum, so its cloud "
                "dimension is pure optimiser scatter; the degenerate control has "
                "exactly 20 flat directions by construction",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(out, open("/root/autodl-tmp/E19_dispersion_null.json", "w"), indent=2)
    print("\n=== E-19 RESULT ===", flush=True)
    print(f"identifiable (no manifold) : cloud dim "
          f"{res['identifiable']['cloud_participation_dim']:.1f} of 126", flush=True)
    print(f"degenerate (20 flat dims)  : cloud dim "
          f"{res['degenerate_k20']['cloud_participation_dim']:.1f} of 126", flush=True)
    print("reported FlyGym            : cloud dim 21.7 of 126", flush=True)
    print("E19_DONE %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
