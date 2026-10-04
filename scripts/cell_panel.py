"""One estimator across a panel of cell models, neural and not.

The single-cell end of the comparison has rested on two models, which is too few
to say anything about cells as a class. This widens it to twelve, split into
those whose output is an action potential and those whose output is a
biochemical oscillation, and measures all of them the same way:

  - parameters are the model's own kinetic or conductance parameters,
    perturbed multiplicatively in log space
  - the observable is the model's own output variable sampled over time, with
    the number of samples pinned to a fixed multiple of the parameter count, so
    that no model is read through fewer observables than another
  - curvature is the Gauss-Newton form used everywhere else in this work

and, where it is affordable, actually recovers the parameters from a randomised
start so that the geometry can be checked against what recovery does.

Nothing here is arranged to come out a particular way. The panel exists because
two points do not make a class.
"""
import json, os, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
import _simguard

CTX = mp.get_context("spawn")
NW = int(os.environ.get("NW", "26"))
OUT = "cell_panel_result.json"
DELTA = 0.05
OBS_PER_PAR = 12
SEEDS = int(os.environ.get("SEEDS", "12"))

# ============================================================ neural cell models

def _hh_gates(V):
    d1 = V + 40.0
    a_m = np.where(np.abs(d1) < 1e-6, 1.0, 0.1 * d1 / (1 - np.exp(-d1 / 10.0)))
    b_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
    a_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    b_h = 1.0 / (1 + np.exp(-(V + 35.0) / 10.0))
    d2 = V + 55.0
    a_n = np.where(np.abs(d2) < 1e-6, 0.1, 0.01 * d2 / (1 - np.exp(-d2 / 10.0)))
    b_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return a_m, b_m, a_h, b_h, a_n, b_n


def hh(t, y, p):
    gNa, gK, gL = p
    V, m, h, n = y
    a_m, b_m, a_h, b_h, a_n, b_n = _hh_gates(V)
    dV = (12.0 - gNa * m**3 * h * (V - 50.0) - gK * n**4 * (V + 77.0)
          - gL * (V + 54.4))
    return [dV, a_m * (1 - m) - b_m * m, a_h * (1 - h) - b_h * h,
            a_n * (1 - n) - b_n * n]


def hh_a(t, y, p):
    """Hodgkin-Huxley with a transient A-type potassium current."""
    gNa, gK, gA, gL = p
    V, m, h, n, a, b = y
    a_m, b_m, a_h, b_h, a_n, b_n = _hh_gates(V)
    a_inf = 1.0 / (1 + np.exp(-(V + 60.0) / 8.5))
    b_inf = 1.0 / (1 + np.exp((V + 78.0) / 6.0))
    dV = (12.0 - gNa * m**3 * h * (V - 50.0) - gK * n**4 * (V + 77.0)
          - gA * a**3 * b * (V + 77.0) - gL * (V + 54.4))
    return [dV, a_m * (1 - m) - b_m * m, a_h * (1 - h) - b_h * h,
            a_n * (1 - n) - b_n * n, (a_inf - a) / 5.0, (b_inf - b) / 25.0]


def wang_buzsaki(t, y, p):
    """Fast-spiking interneuron model (Wang and Buzsaki 1996)."""
    gNa, gK, gL = p
    V, h, n = y
    a_m = -0.1 * (V + 35.0) / (np.exp(-0.1 * (V + 35.0)) - 1 + 1e-12)
    b_m = 4.0 * np.exp(-(V + 60.0) / 18.0)
    m_inf = a_m / (a_m + b_m)
    a_h = 0.07 * np.exp(-(V + 58.0) / 20.0)
    b_h = 1.0 / (np.exp(-0.1 * (V + 28.0)) + 1)
    a_n = -0.01 * (V + 34.0) / (np.exp(-0.1 * (V + 34.0)) - 1 + 1e-12)
    b_n = 0.125 * np.exp(-(V + 44.0) / 80.0)
    dV = (1.0 - gNa * m_inf**3 * h * (V - 55.0) - gK * n**4 * (V + 90.0)
          - gL * (V + 65.0))
    return [dV, 5.0 * (a_h * (1 - h) - b_h * h), 5.0 * (a_n * (1 - n) - b_n * n)]


def morris_lecar(t, y, p):
    gCa, gK, gL, phi = p
    V, N = y
    M_inf = 0.5 * (1 + np.tanh((V + 1.2) / 18.0))
    N_inf = 0.5 * (1 + np.tanh((V - 2.0) / 30.0))
    tau = 1.0 / np.cosh((V - 2.0) / 60.0)
    dV = (90.0 - gCa * M_inf * (V - 120.0) - gK * N * (V + 84.0)
          - gL * (V + 60.0)) / 20.0
    return [dV, phi * (N_inf - N) / tau]


def fitzhugh(t, y, p):
    a, b, tau, I = p
    v, w = y
    return [v - v**3 / 3.0 - w + I, (v + a - b * w) / tau]


def izhikevich(t, y, p):
    """Continuous relaxation, so that the model stays differentiable."""
    a, b, c, d = p
    v, u = y
    dv = 0.04 * v**2 + 5 * v + 140 - u + 10.0
    du = a * (b * v - u)
    reset = 1.0 / (1 + np.exp(-(v - 30.0) / 0.5))
    return [dv - reset * (v - c) * 8.0, du + reset * d * 8.0]


# ======================================================== non-neural cell models

def mapk(t, y, p):
    V1, Ki, V2, k3, k4, V5, V6, k7, k8, V9, V10 = p
    A, Ap, B, Bp, Bpp, C, Cp, Cpp = y
    v1 = V1 * A / ((1.0 + (Cpp / Ki)) * (10.0 + A))
    v2 = V2 * Ap / (8.0 + Ap)
    v3 = k3 * Ap * B / (15.0 + B)
    v4 = k4 * Ap * Bp / (15.0 + Bp)
    v5 = V5 * Bpp / (15.0 + Bpp)
    v6 = V6 * Bp / (15.0 + Bp)
    v7 = k7 * Bpp * C / (15.0 + C)
    v8 = k8 * Bpp * Cp / (15.0 + Cp)
    v9 = V9 * Cpp / (15.0 + Cpp)
    v10 = V10 * Cp / (15.0 + Cp)
    return [v2 - v1, v1 - v2, v6 - v3, v3 + v5 - v4 - v6, v4 - v5,
            v10 - v7, v7 + v9 - v8 - v10, v8 - v9]


def repressilator(t, y, p):
    alpha, alpha0, beta, n = p
    m1, m2, m3, p1, p2, p3 = y
    return [-m1 + alpha / (1 + p3**n) + alpha0,
            -m2 + alpha / (1 + p1**n) + alpha0,
            -m3 + alpha / (1 + p2**n) + alpha0,
            -beta * (p1 - m1), -beta * (p2 - m2), -beta * (p3 - m3)]


def goodwin(t, y, p):
    k1, k2, k3, k4, k5, k6, Ki, n = p
    X, Y, Z = y
    return [k1 * Ki**n / (Ki**n + Z**n) - k2 * X,
            k3 * X - k4 * Y,
            k5 * Y - k6 * Z]


def selkov(t, y, p):
    """Glycolytic oscillator (Selkov 1968)."""
    a, b = p
    x, z = y
    return [-x + a * z + x**2 * z, b - a * z - x**2 * z]


def brusselator(t, y, p):
    A, B = p
    x, z = y
    return [A - (B + 1) * x + x**2 * z, B * x - x**2 * z]


def goldbeter_mitotic(t, y, p):
    """Minimal mitotic oscillator (Goldbeter 1991): cyclin drives cdc2 kinase,
    which activates the protease that degrades cyclin."""
    vi, vd, Kd, kd, VM1, Kc, V2, VM3, V4 = p
    C, M, X = y
    K1 = K2 = K3 = K4 = 0.005
    V1 = VM1 * C / (Kc + C)
    V3 = VM3 * M
    dC = vi - vd * X * C / (Kd + C) - kd * C
    dM = V1 * (1 - M) / (K1 + 1 - M) - V2 * M / (K2 + M)
    dX = V3 * (1 - X) / (K3 + 1 - X) - V4 * X / (K4 + X)
    return [dC, dM, dX]


# ------------------------------------------------------------------- the panel
PANEL = {
    # name: (rhs, y0, theta*, names, output index, t_end, class)
    "HH":            (hh, [-65., .05, .6, .32], [120., 36., .3],
                      ["gNa", "gK", "gL"], 0, 120., "neural"),
    "HH_Acurrent":   (hh_a, [-65., .05, .6, .32, .1, .5], [120., 36., 5., .3],
                      ["gNa", "gK", "gA", "gL"], 0, 120., "neural"),
    "WangBuzsaki":   (wang_buzsaki, [-64., .78, .09], [35., 9., .1],
                      ["gNa", "gK", "gL"], 0, 200., "neural"),
    "MorrisLecar":   (morris_lecar, [-30., .05], [4.4, 8., 2., .04],
                      ["gCa", "gK", "gL", "phi"], 0, 400., "neural"),
    "FitzHughNagumo": (fitzhugh, [-1., 1.], [.7, .8, 12.5, .6],
                       ["a", "b", "tau", "I"], 0, 300., "neural"),
    "MAPK":          (mapk, [90., 10., 240., 45., 15., 240., 45., 15.],
                      [2.5, 9., .25, .025, .025, .75, .75, .025, .025, .5, .5],
                      ["V1", "Ki", "V2", "k3", "k4", "V5", "V6", "k7", "k8",
                       "V9", "V10"], 7, 6000., "biochemical"),
    "Repressilator": (repressilator, [1., 2., 3., 1., 2., 3.],
                      [216., .216, 5., 2.],
                      ["alpha", "alpha0", "beta", "n"], 3, 100., "biochemical"),
    "Goodwin":       (goodwin, [.1, .2, 2.5],
                      [1., .1, 1., .1, 1., .1, 1., 10.],
                      ["k1", "k2", "k3", "k4", "k5", "k6", "Ki", "n"], 2, 200.,
                      "biochemical"),
    "Selkov":        (selkov, [1., 1.], [.1, .6], ["a", "b"], 0, 400.,
                      "biochemical"),
    "Brusselator":   (brusselator, [1., 1.], [1., 3.], ["A", "B"], 0, 200.,
                      "biochemical"),
    "GoldbeterMitotic": (goldbeter_mitotic, [.01, .01, .01],
                         [.025, .25, .02, .01, 3.0, .5, 1.5, 1.0, .5],
                         ["vi", "vd", "Kd", "kd", "VM1", "Kc", "V2", "VM3",
                          "V4"], 0, 120., "biochemical"),
}


def _guarded(f):
    def g(t, y, p):
        _simguard.check()
        return f(t, y, p)
    return g


def run_model(name, theta):
    f, y0, _, _, oi, T, _ = PANEL[name]
    te = np.linspace(0, T, 4000)
    _simguard.start(20.0)
    try:
        sol = solve_ivp(_guarded(f), (0, T), y0,
                        args=(np.asarray(theta, float),),
                        t_eval=te, method="LSODA", rtol=1e-6, atol=1e-9)
    except Exception:
        return None
    if not sol.success or sol.y.shape[1] < len(te) \
            or not np.all(np.isfinite(sol.y)):
        return None
    return sol.y[oi]


# How much of each trace is transient. Chosen per model by stationarity of the
# quarter ranges (cell_stationary.py): the A-current neuron falls silent
# partway through its window and resumes, so a uniform 0.4 begins reading inside
# the silent stretch. Every other model is already stationary at 0.4, so their
# numbers are unchanged by construction.
DROP = {"HH_Acurrent": 0.5}


def observe(name, theta, npts):
    x = run_model(name, theta)
    if x is None:
        return None
    seg = x[int(DROP.get(name, 0.4) * len(x)):]
    idx = np.linspace(0, len(seg) - 1, npts).astype(int)
    return seg[idx].astype(float)


def _col(arg):
    name, k, npts = arg
    star = np.array(PANEL[name][2], dtype=float)
    tp, tm = star.copy(), star.copy()
    tp[k] *= np.exp(DELTA)
    tm[k] *= np.exp(-DELTA)
    a, b = observe(name, tp, npts), observe(name, tm, npts)
    if a is None or b is None:
        return name, k, None
    return name, k, ((a - b) / (2 * DELTA)).astype(np.float64)


def _rec(arg):
    name, seed, npts = arg
    import cma
    star = np.array(PANEL[name][2], dtype=float)
    ob_star = observe(name, star, npts)
    if ob_star is None:
        return None
    ls = np.log(star)

    def f(z):
        z = np.clip(z, ls - 1.5, ls + 1.5)
        ob = observe(name, np.exp(z), npts)
        if ob is None:
            return 5.0
        sc = np.maximum(np.abs(ob_star).mean(), 1e-9)
        return float(np.mean(np.abs(ob - ob_star)) / sc)

    r = np.random.default_rng(seed)
    z0 = ls + r.normal(0, 0.30, size=len(star))
    es = cma.CMAEvolutionStrategy(z0, 0.3,
                                  {"popsize": 12, "maxiter": 70,
                                   "seed": int(seed) + 1, "verbose": -9,
                                   "bounds": [(ls - 1.5).tolist(),
                                              (ls + 1.5).tolist()]})
    es.optimize(f)
    th = np.exp(np.clip(es.result.xbest, ls - 1.5, ls + 1.5))
    return dict(name=name, seed=int(seed), L0=round(f(z0), 5),
                L_final=round(f(np.log(th)), 5),
                rel=round(float(np.linalg.norm(th - star) / np.linalg.norm(star)), 4),
                ratio=(th / star).tolist())


def spectrum(H):
    ev = np.sort(np.maximum(np.linalg.eigvalsh(H), 0))[::-1]
    if ev.sum() <= 0:
        return None
    cs = np.cumsum(ev) / ev.sum()
    pos = ev[ev > ev.max() * 1e-14]
    return {"eff_dim_90": int(np.searchsorted(cs, 0.90) + 1),
            "eff_dim_99": int(np.searchsorted(cs, 0.99) + 1),
            "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "spectrum_decades": float(np.log10(pos.max() / pos.min()))
            if len(pos) > 1 else 0.0}


def main():
    t0 = time.time()
    out = {"estimator": "Gauss-Newton H = J^T W J, central differences in log "
                        "parameters", "obs_per_param": OBS_PER_PAR,
           "delta_log": DELTA, "models": {}}

    live = []
    for name in PANEL:
        star = np.array(PANEL[name][2], float)
        npts = OBS_PER_PAR * len(star)
        ob = observe(name, star, npts)
        if ob is None or np.ptp(ob) <= 0:
            print(f"  {name}: baseline not usable, skipped", flush=True)
            continue
        # the observable must still be moving at the end of the window: a model
        # that has settled onto a fixed point has no behaviour to measure
        half = len(ob) // 2
        early, late = np.ptp(ob[:half]), np.ptp(ob[half:])
        if late < 0.2 * max(early, 1e-30) or late <= 1e-9 * max(abs(ob).max(), 1e-30):
            print(f"  {name}: output settles to a fixed point "
                  f"(late/early range {late/max(early,1e-30):.2g}), skipped",
                  flush=True)
            continue
        live.append((name, npts))
    print(f"{len(live)} of {len(PANEL)} models usable\n", flush=True)

    jobs = [(n, k, npts) for n, npts in live
            for k in range(len(PANEL[n][2]))]
    cols = {}
    with ProcessPoolExecutor(max_workers=NW, mp_context=CTX) as pool:
        for name, k, v in pool.map(_col, jobs, chunksize=1):
            cols[(name, k)] = v

    print("curvature")
    for name, npts in live:
        star = PANEL[name][2]
        npar = len(star)
        good = [k for k in range(npar) if cols.get((name, k)) is not None]
        if not good:
            continue
        L = min(len(cols[(name, k)]) for k in good)
        J = np.array([cols[(name, k)][:L] for k in good]).T
        sd = J.std(axis=1)
        keep = sd > 1e-12 * max(sd.max(), 1e-30)
        Jw = (J[keep] / sd[keep][:, None]) if keep.any() else J
        sp = spectrum(Jw.T @ Jw)
        if sp is None:
            continue
        rec = {"class": PANEL[name][6], "n_params": npar,
               "n_obs": int(Jw.shape[0]),
               "obs_per_param": round(Jw.shape[0] / npar, 1),
               "n_dead_params": npar - len(good),
               "param_names": PANEL[name][3],
               "elasticity": {n: float(v) for n, v in
                              zip(PANEL[name][3], np.linalg.norm(J, axis=0))},
               **sp,
               "constrained_fraction_90": sp["eff_dim_90"] / npar,
               "constrained_fraction_99": sp["eff_dim_99"] / npar}
        out["models"][name] = rec
        print(f"  {name:16s} {rec['class']:12s} par {npar:3d} obs {rec['n_obs']:4d}"
              f"  eff {sp['eff_dim_90']:2d}/{sp['eff_dim_99']:2d}"
              f"  frac90 {rec['constrained_fraction_90']:.3f}"
              f"  PR {sp['participation_ratio']:5.2f}"
              f"  {sp['spectrum_decades']:5.1f}dec", flush=True)

    print(f"\nrecovery, {SEEDS} seeds per model", flush=True)
    rjobs = [(n, s, npts) for n, npts in live for s in range(SEEDS)]
    res = [r for r in
           ProcessPoolExecutor(max_workers=NW, mp_context=CTX).map(_rec, rjobs,
                                                                  chunksize=1)
           if r]
    for name, npts in live:
        g = [r for r in res if r["name"] == name]
        if not g or name not in out["models"]:
            continue
        rel = np.array([r["rel"] for r in g])
        Lf = np.array([r["L_final"] for r in g])
        R = np.array([r["ratio"] for r in g])
        out["models"][name]["recovery"] = {
            "n_seeds": len(g),
            "rel_theta_median": float(np.median(rel)),
            "rel_theta_iqr": [float(np.percentile(rel, 25)),
                              float(np.percentile(rel, 75))],
            "L_final_median": float(np.median(Lf)),
            "max_excursion_x": float(R.max()),
            "per_param_median_ratio": {n: float(v) for n, v in
                                       zip(PANEL[name][3], np.median(R, 0))}}
        print(f"  {name:16s} behaviour {np.median(Lf):.5f}  "
              f"param error {np.median(rel):.3f}  max {R.max():.1f}x", flush=True)

    # does per-parameter sensitivity predict which parameters recovery gets right?
    from scipy.stats import spearmanr
    for name, m in out["models"].items():
        if "recovery" not in m or m["n_params"] < 3:
            continue
        nm = m["param_names"]
        el = np.array([m["elasticity"][n] for n in nm])
        err = np.array([abs(m["recovery"]["per_param_median_ratio"][n] - 1.0)
                        for n in nm])
        rho, p = spearmanr(el, err)
        m["sensitivity_vs_error"] = {"spearman_rho": float(rho), "p": float(p),
                                     "n": len(nm)}
    pooled = [(m["elasticity"][n] /
               max(m["elasticity"].values()),
               abs(m["recovery"]["per_param_median_ratio"][n] - 1.0))
              for m in out["models"].values() if "recovery" in m
              for n in m["param_names"]]
    if len(pooled) > 5:
        a = np.array(pooled)
        rho, p = spearmanr(a[:, 0], a[:, 1])
        out["pooled_sensitivity_vs_error"] = {"spearman_rho": float(rho),
                                              "p": float(p), "n": len(pooled)}
        print(f"\npooled across models: relative sensitivity vs recovery error "
              f"rho {rho:+.3f}  p {p:.2g}  n {len(pooled)}", flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved -> {OUT}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
