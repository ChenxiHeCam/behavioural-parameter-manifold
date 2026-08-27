"""Why does our Hodgkin-Huxley recovery succeed where the literature reports
degeneracy?

Golowasch (2002), Achard and De Schutter (2006) and the Prinz-Marder line report
that conductance-based neurons are badly degenerate: averaging conductances gives
the wrong behaviour, and wide regions of parameter space produce the same trace.
Our run recovers g_Na and g_K to a few per cent. Both cannot be general
statements about "the HH model", so the difference must lie in the conditions.

Two candidates, and this measures both at once:

  how many parameters are free   3  (conductances only, kinetics fixed at truth)
                                 5  (+ leak reversal, capacitance)
                                 8  (+ scalings on the m, h, n rate functions)

  how much is recorded           1    spike count at one current
                                 7    F-I curve over five currents + spike shape
                                 ~200 the voltage trace at one current
                                 ~1000 traces at all five currents

The literature sits at many parameters matched against one trace; we sat at few
parameters against a multi-current assay. If that is the whole explanation, then
identifiability should fall along the parameter axis and rise along the observable
axis, and our number should be recoverable as one corner of the same table.

Reported per cell: the Gauss-Newton effective dimension, the spread of the
eigenvalue spectrum in decades (the Gutenkunst sloppiness signature), and the
parameter error of an actual recovery from a randomised start.
"""
import json, os, time, itertools
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
import _simguard

CTX = mp.get_context("spawn")
NW = 5
SEEDS = 8
OUT = "cell_ladder_result.json"

E_Na, E_K = 50.0, -77.0
I_LEVELS = np.array([4.0, 7.0, 10.0, 15.0, 20.0])

# full parameter vector; a setting uses the first n of these
FULL_NAMES = ["g_Na", "g_K", "g_L", "E_L", "C_m", "s_m", "s_h", "s_n"]
FULL_STAR = np.array([120.0, 36.0, 0.3, 54.4, 1.0, 1.0, 1.0, 1.0])
# E_L is stored positive and negated on use, so every parameter is positive and
# the search can run in log space throughout


def _unpack(th):
    v = FULL_STAR.copy()
    v[:len(th)] = th
    gNa, gK, gL, EL, Cm, sm, sh, sn = v
    return gNa, gK, gL, -EL, Cm, sm, sh, sn


def rhs(t, y, th, I):
    _simguard.check()
    gNa, gK, gL, E_L, C_m, sm, sh, sn = _unpack(th)
    V, m, h, n = y
    d1 = V + 40.0
    a_m = 1.0 if abs(d1) < 1e-6 else 0.1 * d1 / (1 - np.exp(-d1 / 10.0))
    b_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
    a_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    b_h = 1.0 / (1 + np.exp(-(V + 35.0) / 10.0))
    d2 = V + 55.0
    a_n = 0.1 if abs(d2) < 1e-6 else 0.01 * d2 / (1 - np.exp(-d2 / 10.0))
    b_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
    dV = (I - gNa * m**3 * h * (V - E_Na) - gK * n**4 * (V - E_K) - gL * (V - E_L)) / C_m
    return [dV, sm * (a_m * (1 - m) - b_m * m),
            sh * (a_h * (1 - h) - b_h * h),
            sn * (a_n * (1 - n) - b_n * n)]


def _diverged(t, y, th, I):
    return 500.0 - abs(y[0])
_diverged.terminal = True
_diverged.direction = -1


def _trace(th, I, T=80.0, dt=0.02):
    te = np.arange(0, T, dt)
    _simguard.start(3.0)
    try:
        sol = solve_ivp(rhs, (0, T), [-65.0, 0.05, 0.6, 0.32], args=(th, I),
                        t_eval=te, method="RK45", rtol=1e-5, atol=1e-7,
                        max_step=0.1, events=_diverged)
    except _simguard.Budget:
        return te, np.full(len(te), np.nan)
    if sol.y.shape[1] < len(te):
        return te, np.full(len(te), np.nan)
    return sol.t, sol.y[0]


def _spikes(t, V, thr=0.0):
    return int(((V[:-1] < thr) & (V[1:] >= thr)).sum())


def obs_single(th):
    t, V = _trace(th, 15.0)
    if not np.all(np.isfinite(V)):
        return None
    return np.array([_spikes(t, V) / (t[-1] / 1000.0)])


def obs_fi7(th):
    rates = []
    for I in I_LEVELS:
        t, V = _trace(th, I)
        if not np.all(np.isfinite(V)):
            return None
        rates.append(_spikes(t, V) / (t[-1] / 1000.0))
    t, V = _trace(th, 15.0)
    amp = float(V.max() - V.min())
    hw = float((V > (V.min() + 0.5 * amp)).sum() * (t[1] - t[0]))
    return np.array(rates + [amp, hw])


def obs_trace1(th):
    t, V = _trace(th, 15.0)
    if not np.all(np.isfinite(V)):
        return None
    return V[::20].astype(float)


def obs_trace5(th):
    out = []
    for I in I_LEVELS:
        t, V = _trace(th, I)
        if not np.all(np.isfinite(V)):
            return None
        out.append(V[::20])
    return np.concatenate(out).astype(float)


OBS = {"single": obs_single, "fi7": obs_fi7, "trace1": obs_trace1, "trace5": obs_trace5}
NPAR = [3, 5, 8]


def loss(fn, ob_star, th):
    ob = fn(th)
    if ob is None or len(ob) != len(ob_star):
        return 5.0
    sc = np.maximum(np.abs(ob_star), 1e-3)
    return float(np.mean(np.abs(ob - ob_star) / sc))


def curvature(okey, npar, delta=0.05):
    fn = OBS[okey]
    star = FULL_STAR[:npar]
    ob0 = fn(star)
    if ob0 is None:
        return None
    ob0 = np.asarray(ob0, float)
    w = 1.0 / np.maximum(np.abs(ob0), 1e-3)
    cols = []
    for k in range(npar):
        tp, tm = star.copy(), star.copy()
        tp[k] *= np.exp(delta)
        tm[k] *= np.exp(-delta)
        a, b = fn(tp), fn(tm)
        if a is None or b is None:
            cols.append(np.zeros_like(ob0))
            continue
        cols.append((np.asarray(a, float) - np.asarray(b, float)) / (2 * delta) * w)
    J = np.array(cols).T
    H = J.T @ J
    ev = np.sort(np.maximum(np.linalg.eigvalsh(H), 0))[::-1]
    if ev.sum() <= 0:
        return {"n_obs": len(ob0), "eff_dim_90": 0, "eff_dim_99": 0,
                "participation_ratio": 0.0, "spectrum_decades": 0.0}
    cs = np.cumsum(ev) / ev.sum()
    pos = ev[ev > ev.max() * 1e-14]
    return {"n_obs": int(len(ob0)),
            "eff_dim_90": int(np.searchsorted(cs, 0.90) + 1),
            "eff_dim_99": int(np.searchsorted(cs, 0.99) + 1),
            "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "spectrum_decades": float(np.log10(pos.max() / pos.min())) if len(pos) > 1 else 0.0,
            "elasticity": {n: float(v) for n, v in zip(FULL_NAMES[:npar],
                                                       np.linalg.norm(J, axis=0))}}


def recover(arg):
    okey, npar, seed = arg
    import cma
    fn = OBS[okey]
    star = FULL_STAR[:npar]
    ob_star = fn(star)
    if ob_star is None:
        return None
    ob_star = np.asarray(ob_star, float)
    logstar = np.log(star)
    r = np.random.default_rng(seed)
    z0 = logstar + r.normal(0, 0.30, size=npar)
    f = lambda z: loss(fn, ob_star, np.exp(np.clip(z, logstar - 2.0, logstar + 2.0)))
    es = cma.CMAEvolutionStrategy(z0, 0.3,
                                  {"popsize": 12, "maxiter": 60, "seed": int(seed) + 1,
                                   "verbose": -9,
                                   "bounds": [(logstar - 2.0).tolist(),
                                              (logstar + 2.0).tolist()]})
    es.optimize(f)
    th = np.exp(np.clip(es.result.xbest, logstar - 2.0, logstar + 2.0))
    return dict(okey=okey, npar=npar, seed=int(seed), L0=round(f(z0), 4),
                L_final=round(f(np.log(th)), 5),
                rel=round(float(np.linalg.norm(th - star) / np.linalg.norm(star)), 4),
                ratio=(th / star).tolist())


def main():
    t0 = time.time()
    out = {"question": "why does 3-parameter HH recover where the literature reports "
                       "degeneracy?", "grid": {}}

    print("curvature grid  (n_obs / eff dim 90 / spectrum decades)")
    for npar in NPAR:
        row = []
        for okey in OBS:
            c = curvature(okey, npar)
            out["grid"][f"{okey}_p{npar}"] = {"curvature": c}
            row.append(f"{okey}: {c['n_obs']:4d}obs {c['eff_dim_90']}/{npar} "
                       f"{c['spectrum_decades']:.1f}dec")
        print(f"  {npar} params | " + " | ".join(row), flush=True)

    jobs = [(o, p, s) for o, p, s in itertools.product(OBS, NPAR, range(SEEDS))]
    print(f"\n{len(jobs)} recoveries on {NW} workers", flush=True)
    res = []
    with ProcessPoolExecutor(max_workers=NW, mp_context=CTX) as pool:
        for n, r in enumerate(pool.map(recover, jobs, chunksize=1), 1):
            if r:
                res.append(r)
            if n % 24 == 0:
                print(f"  {n}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    print("\nrecovery grid  (median parameter error | median behaviour distance)")
    for npar in NPAR:
        cells = []
        for okey in OBS:
            g = [r for r in res if r["okey"] == okey and r["npar"] == npar]
            if not g:
                continue
            rel = np.array([r["rel"] for r in g])
            Lf = np.array([r["L_final"] for r in g])
            R = np.array([r["ratio"] for r in g])
            rec = {"n_seeds": len(g),
                   "rel_theta_median": float(np.median(rel)),
                   "rel_theta_iqr": [float(np.percentile(rel, 25)),
                                     float(np.percentile(rel, 75))],
                   "L_final_median": float(np.median(Lf)),
                   "max_excursion_x": float(R.max()),
                   "per_param_median_ratio": {n: float(v) for n, v in
                                              zip(FULL_NAMES[:npar], np.median(R, 0))}}
            out["grid"][f"{okey}_p{npar}"]["recovery"] = rec
            cells.append(f"{okey}: {np.median(rel):.2f} | {np.median(Lf):.4f}")
        print(f"  {npar} params | " + " | ".join(cells), flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved -> {OUT}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
