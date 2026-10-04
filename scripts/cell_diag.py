"""Why do two of the neural cell models almost never converge?

In the panel run Hodgkin-Huxley matched the behaviour on 23 of 24 random starts
and Morris-Lecar on 17, while Wang-Buzsaki managed 2 and the A-current neuron 1.
All four are three- or four-parameter conductance models, so the difference is
not the size of the search. Until it is explained, the "recovered exactly" entry
for those two rests on one and two seeds and cannot be reported alongside the
others.

Three explanations are separable and all three are measured here.

  the threshold      the fits may be landing just outside the 0.02 cut, in which
                     case the models are fine and the gate is too tight; the full
                     distribution of final losses answers this directly

  the window         the observation window holds only two or three oscillation
                     cycles, and the amplitude quantiles, spectrum and
                     autocorrelation are all estimated from those few cycles. A
                     four-fold longer window is run as a second arm. The A-current
                     neuron additionally falls silent partway through its default
                     window, so its observable straddles a gap

  the landscape      if the loss barely rises as the parameters move away from
                     the truth, no optimiser will find it. A distance-to-loss
                     profile is measured for each model and reported alongside
"""
import json, os, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import cell_panel as P
import cell_panel3 as P3

CTX = mp.get_context("spawn")
NW = int(os.environ.get("NW", "26"))
SEEDS = int(os.environ.get("SEEDS", "32"))
MODELS = ["HH", "MorrisLecar", "WangBuzsaki", "HH_Acurrent"]
FOLD = 3.0
OUT = "cell_diag_result.json"


def _observe(name, theta, npts, tmul):
    """Observation with the model's window optionally lengthened."""
    if tmul == 1:
        return P3.phase_invariant(name, theta, npts)
    entry = list(P.PANEL[name])
    entry[5] = entry[5] * tmul
    old = P.PANEL[name]
    P.PANEL[name] = tuple(entry)
    try:
        return P3.phase_invariant(name, theta, npts)
    finally:
        P.PANEL[name] = old


def _profile(arg):
    """How much does the loss rise as the parameters move away from the truth?"""
    name, tmul = arg
    star = np.array(P.PANEL[name][2], float)
    d = len(star)
    npts = P.OBS_PER_PAR * d
    ob = _observe(name, star, npts, tmul)
    if ob is None:
        return name, tmul, None
    sc = max(np.abs(ob).mean(), 1e-9)
    r = np.random.default_rng(0)
    out = {}
    for fold in (1.05, 1.2, 1.5, 2.0, 3.0, 10.0):
        vals = []
        for _ in range(12):
            th = star * np.exp(r.uniform(-np.log(fold), np.log(fold), size=d))
            o = _observe(name, th, npts, tmul)
            vals.append(5.0 if o is None or len(o) != len(ob)
                        else float(np.mean(np.abs(o - ob)) / sc))
        out[str(fold)] = {"median_loss": float(np.median(vals)),
                          "min_loss": float(np.min(vals))}
    return name, tmul, out


def _fit(arg):
    name, tmul, seed = arg
    import cma
    star = np.array(P.PANEL[name][2], float)
    d = len(star)
    npts = P.OBS_PER_PAR * d
    ob = _observe(name, star, npts, tmul)
    if ob is None:
        return None
    sc = max(np.abs(ob).mean(), 1e-9)
    ls = np.log(star)
    lf = np.log(FOLD)
    lo, hi = ls - (lf + 1.0), ls + (lf + 1.0)
    rng = np.random.default_rng(seed * 977 + int(tmul))
    z0 = ls + rng.uniform(-lf, lf, size=d)

    def f(z):
        o = _observe(name, np.exp(np.clip(z, lo, hi)), npts, tmul)
        if o is None or len(o) != len(ob):
            return 5.0
        return float(np.mean(np.abs(o - ob)) / sc)

    es = cma.CMAEvolutionStrategy(
        z0.copy(), 0.5 * lf,
        {"popsize": 16, "maxiter": 400, "seed": seed * 31 + 7, "verbose": -9,
         "tolfun": 1e-13, "tolx": 1e-12, "bounds": [lo.tolist(), hi.tolist()]})
    es.optimize(f)
    z = np.clip(es.result.xbest, lo, hi)
    th = np.exp(z)
    return dict(name=name, tmul=tmul, seed=int(seed), L_final=float(f(z)),
                rel_after=float(np.linalg.norm(th - star) / np.linalg.norm(star)),
                rel_before=float(np.linalg.norm(np.exp(z0) - star) /
                                 np.linalg.norm(star)))


def main():
    t0 = time.time()
    out = {"models": MODELS, "seeds": SEEDS, "fold": FOLD,
           "loss_profile": {}, "fits": {}}

    print("loss against distance from the truth (median over 12 draws)")
    jobs = [(n, t) for n in MODELS for t in (1, 4)]
    with ProcessPoolExecutor(max_workers=min(NW, len(jobs)),
                             mp_context=CTX) as pool:
        for name, tmul, prof in pool.map(_profile, jobs):
            if prof is None:
                continue
            out["loss_profile"][f"{name}_T{tmul}x"] = prof
            row = "  ".join(f"{k}x:{v['median_loss']:.3f}" for k, v in prof.items())
            print(f"  {name:14s} window {tmul}x   {row}", flush=True)

    print(f"\n{SEEDS} random starts per model per window, 1/3-3x")
    fjobs = [(n, t, s) for n in MODELS for t in (1, 4) for s in range(SEEDS)]
    res = []
    with ProcessPoolExecutor(max_workers=NW, mp_context=CTX) as pool:
        for r in pool.map(_fit, fjobs, chunksize=1):
            if r:
                res.append(r)

    print("\nfinal loss distribution and what each cut-off would give")
    for n in MODELS:
        for t in (1, 4):
            g = [r for r in res if r["name"] == n and r["tmul"] == t]
            if not g:
                continue
            L = np.array([r["L_final"] for r in g])
            rel = np.array([r["rel_after"] for r in g])
            rec = {"n": len(g),
                   "L_median": float(np.median(L)), "L_min": float(L.min()),
                   "L_quartiles": [float(np.percentile(L, q)) for q in (25, 50, 75)],
                   "frac_below_0.02": float((L < 0.02).mean()),
                   "frac_below_0.05": float((L < 0.05).mean()),
                   "frac_below_0.10": float((L < 0.10).mean()),
                   "rel_median_below_0.05": float(np.median(rel[L < 0.05]))
                   if (L < 0.05).any() else None}
            out["fits"][f"{n}_T{t}x"] = rec
            print(f"  {n:14s} window {t}x  L med {np.median(L):.4f} "
                  f"min {L.min():.5f}  <0.02 {rec['frac_below_0.02']*100:3.0f}%  "
                  f"<0.05 {rec['frac_below_0.05']*100:3.0f}%  "
                  f"<0.10 {rec['frac_below_0.10']*100:3.0f}%  "
                  f"param err {rec['rel_median_below_0.05']}", flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved -> {OUT}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
