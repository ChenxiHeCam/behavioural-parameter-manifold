"""Cell-scale PGOB: what the existing runs actually show, and the two things they
cannot show as they stand.

Two corrections drive this script.

  1. The MAPK run reads the cell through FIVE scalar features while searching
     ELEVEN parameters. A Gauss-Newton curvature built from n_obs observables has
     rank at most n_obs, so that run cannot distinguish intrinsic degeneracy from
     a measurement that was never able to see eleven directions. This is the same
     correction that moved FlyGym from 1 to 16. We therefore re-run it against the
     MAPK-PP trajectory itself, where observables outnumber parameters, and
     compare.

  2. The prior-box run reports that 100% of recovered parameters land within
     1/3--3x of canonical. That is the optimiser's own hard bound restated, not a
     finding. The quantity that is a finding is what the box costs in behaviour
     and what it buys in parameter error.

The third measurement is the one the whole-organism models structurally cannot
make: with ground truth known, does the per-parameter sensitivity predict which
parameters recovery actually gets right?
"""
import json, os, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import pgob_cell_hh as HH
import pgob_cell_mapk as MK

CTX = mp.get_context("spawn")
NW = 5
SEEDS = 20
OUT = "pgob_cell_audit_result.json"


# ----------------------------------------------------------------- observables
def mapk_rich(th):
    """The MAPK-PP trajectory itself, subsampled: 120 observables for 11 parameters."""
    t, x = MK.simulate(th)
    if not np.all(np.isfinite(x)):
        return None
    seg = x[int(0.4 * len(x)):]
    return seg[::5][:120].astype(float)


SPEC = {
    "HH_spike":    dict(mod=HH, obs=HH.observables, star=HH.THETA_STAR,
                        names=HH.THETA_NAMES, sigma=0.35),
    "MAPK_scalar": dict(mod=MK, obs=MK.observables, star=MK.THETA_STAR,
                        names=MK.THETA_NAMES, sigma=0.40),
    "MAPK_rich":   dict(mod=MK, obs=mapk_rich, star=MK.THETA_STAR,
                        names=MK.THETA_NAMES, sigma=0.40),
}


def _loss(obs_fn, ob_star, th):
    ob = obs_fn(th)
    if ob is None or len(ob) != len(ob_star):
        return 5.0
    return float(np.mean(np.abs(ob - ob_star) / (np.abs(ob_star) + 1e-9)))


# ------------------------------------------------------------------- curvature
def jacobian(key, delta=0.05):
    """Central-difference observable Jacobian in log-parameter space, whitened."""
    s = SPEC[key]
    star, obs = s["star"], s["obs"]
    ob0 = np.asarray(obs(star), dtype=float)
    w = 1.0 / (np.abs(ob0) + 1e-9)
    cols = []
    for k in range(len(star)):
        tp, tm = star.copy(), star.copy()
        tp[k] *= np.exp(delta)
        tm[k] *= np.exp(-delta)
        a, b = obs(tp), obs(tm)
        if a is None or b is None:
            cols.append(np.zeros_like(ob0))
            continue
        cols.append((np.asarray(a, float) - np.asarray(b, float)) / (2 * delta) * w)
    return np.array(cols).T, ob0


def eff_dim(H):
    ev = np.sort(np.maximum(np.linalg.eigvalsh(H), 0))[::-1]
    if ev.sum() <= 0:
        return 0, 0, 0.0
    cs = np.cumsum(ev) / ev.sum()
    return (int(np.searchsorted(cs, 0.90) + 1), int(np.searchsorted(cs, 0.99) + 1),
            float(ev.sum() ** 2 / (ev ** 2).sum()))


# -------------------------------------------------------------------- recovery
def _one(arg):
    key, seed, box = arg
    import cma
    s = SPEC[key]
    star, obs = s["star"], s["obs"]
    ob_star = np.asarray(obs(star), dtype=float)
    logstar = np.log(star)
    r = np.random.default_rng(seed)

    if box is None:
        z0 = logstar + r.normal(0, s["sigma"], size=len(star))
        bounds = None
    else:
        lb, ub = logstar - np.log(box), logstar + np.log(box)
        z0 = logstar + r.uniform(-np.log(box), np.log(box), size=len(star))
        bounds = [lb.tolist(), ub.tolist()]

    def f(z):
        if bounds is not None:
            z = np.clip(z, bounds[0], bounds[1])
        return _loss(obs, ob_star, np.exp(z))

    opt = {"popsize": 16, "maxiter": 90, "seed": int(seed) + 1, "verbose": -9}
    if bounds:
        opt["bounds"] = bounds
    es = cma.CMAEvolutionStrategy(z0, 0.4, opt)
    es.optimize(f)
    z = es.result.xbest
    if bounds is not None:
        z = np.clip(z, bounds[0], bounds[1])
    th = np.exp(z)
    return dict(key=key, seed=int(seed), box=box, L0=round(f(z0), 4),
                L_final=round(f(z), 5),
                rel=round(float(np.linalg.norm(th - star) / np.linalg.norm(star)), 4),
                ratio=(th / star).tolist())


def main():
    t0 = time.time()
    out = {"note": "cell-scale PGOB, ground truth known", "runs": {}}

    # --------------------------------------------- curvature, all three settings
    print("curvature (Gauss-Newton, log-parameter, whitened)")
    for key in SPEC:
        J, ob0 = jacobian(key)
        H = J.T @ J
        k90, k99, pr = eff_dim(H)
        el = np.linalg.norm(J, axis=0)
        s = SPEC[key]
        out["runs"].setdefault(key, {})["curvature"] = {
            "n_obs": int(len(ob0)), "n_params": int(len(s["star"])),
            "obs_exceed_params": bool(len(ob0) > len(s["star"])),
            "eff_dim_90": k90, "eff_dim_99": k99, "participation_ratio": pr,
            "elasticity": {n: float(v) for n, v in zip(s["names"], el)},
        }
        print(f"  {key:12s} n_obs {len(ob0):4d} vs n_par {len(s['star']):3d}  "
              f"eff dim {k90}/{k99} of {len(s['star'])}  PR {pr:.2f}"
              f"{'' if len(ob0) > len(s['star']) else '   <-- rank-capped by the assay'}")

    # ------------------------------------------------------------ recovery runs
    jobs = ([("HH_spike", s, None) for s in range(SEEDS)] +
            [("MAPK_scalar", s, None) for s in range(SEEDS)] +
            [("MAPK_rich", s, None) for s in range(SEEDS)] +
            [("MAPK_scalar", s, 3.0) for s in range(SEEDS)] +
            [("MAPK_rich", s, 3.0) for s in range(SEEDS)])
    print(f"\n{len(jobs)} recoveries on {NW} workers")
    res = []
    with ProcessPoolExecutor(max_workers=NW, mp_context=CTX) as pool:
        for n, r in enumerate(pool.map(_one, jobs, chunksize=1), 1):
            res.append(r)
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    for key in SPEC:
        for box in (None, 3.0):
            g = [r for r in res if r["key"] == key and r["box"] == box]
            if not g:
                continue
            R = np.array([r["ratio"] for r in g])
            rel = np.array([r["rel"] for r in g])
            Lf = np.array([r["L_final"] for r in g])
            tag = "free" if box is None else f"box{box:g}"
            rec = {"n_seeds": len(g),
                   "rel_theta_median": float(np.median(rel)),
                   "rel_theta_iqr": [float(np.percentile(rel, 25)),
                                     float(np.percentile(rel, 75))],
                   "L_final_median": float(np.median(Lf)),
                   "L_final_max": float(Lf.max()),
                   "max_excursion_x": float(R.max()),
                   "min_excursion_x": float(R.min()),
                   "frac_within_3x": float(((R >= 1 / 3) & (R <= 3)).mean()),
                   "per_param_median_ratio": {n: float(v) for n, v in
                                              zip(SPEC[key]["names"], np.median(R, 0))}}
            out["runs"][key][tag] = rec
            print(f"  {key:12s} {tag:6s}  behaviour {np.median(Lf):.5f}  "
                  f"rel_theta {np.median(rel):.3f}  max {R.max():.1f}x  "
                  f"within3x {rec['frac_within_3x']*100:.0f}%")

    # ------------------- does sensitivity predict which parameters get recovered?
    from scipy.stats import spearmanr
    for key in SPEC:
        c = out["runs"][key]["curvature"]
        f = out["runs"][key].get("free")
        if not f:
            continue
        names = SPEC[key]["names"]
        el = np.array([c["elasticity"][n] for n in names])
        err = np.array([abs(f["per_param_median_ratio"][n] - 1.0) for n in names])
        if len(names) < 3:
            continue
        rho, p = spearmanr(el, err)
        out["runs"][key]["sensitivity_predicts_error"] = {
            "spearman_rho": float(rho), "p": float(p), "n": len(names),
            "worst_recovered": names[int(np.argmax(err))],
            "least_sensitive": names[int(np.argmin(el))]}
        print(f"  {key:12s} elasticity vs recovery error: rho {rho:+.2f} p {p:.3f}"
              f"   worst={names[int(np.argmax(err))]} least-sensitive={names[int(np.argmin(el))]}")

    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved -> {OUT}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
