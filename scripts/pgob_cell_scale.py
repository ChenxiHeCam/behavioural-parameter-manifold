"""Does the fraction of parameter directions that behaviour constrains depend on
how many cells there are?

The comparison across published simulators cannot answer this, because every
model differs in three ways at once: how many parameters it has, which kind of
parameter they are, and how much of its output was recorded. FlyGym makes that
visible -- 48 biomechanical gains give a higher constrained fraction than an
11-parameter signalling cascade -- and flyvis makes it visible within a single
model, where 65 cell-type biases and 65 membrane time constants give 0.66 and
0.08.

So this holds two of the three fixed. One cell model throughout (Hodgkin-Huxley),
one parameter type throughout (the three ionic conductances of each cell), fixed
synaptic wiring that is never searched, and an observable count pinned to a
constant multiple of the parameter count. The only thing that varies is how many
cells are in the network:

    N = 1, 2, 4, 8, 16, 32, 64   ->   3, 6, 12, 24, 48, 96, 192 parameters

spanning the gap between the single-cell models and the connectome-scale ones.
Reported per size: the Gauss-Newton effective dimension at 90% and 99%, the
threshold-free participation ratio, the constrained fraction, and the spread of
the eigenvalue spectrum in decades.

No hypothesis is being defended here. If the fraction falls with size, that is a
size effect; if it does not, the ordering seen across the published simulators is
about something other than size.
"""
import json, os, time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import solve_ivp
import _simguard

CTX = mp.get_context("spawn")
NW = int(os.environ.get("NW", "26"))
OUT = "pgob_cell_scale_result.json"

E_Na, E_K, E_L, C_m = 50.0, -77.0, -54.4, 1.0
E_SYN = 0.0            # excitatory reversal
G_SYN = 0.10           # fixed synaptic strength; never searched
V_HALF, V_SLOPE = -20.0, 5.0
I_DRIVE = 12.0
T_END, DT = 120.0, 0.02
G_STAR = np.array([120.0, 36.0, 0.3])          # per-cell conductances
SIZES = [1, 2, 4, 8, 16, 32, 64]
OBS_PER_PAR = [10, 30]                          # observable budget, held fixed
DELTA = 0.05


def wiring(N, seed=0):
    """Fixed sparse directed wiring, identical for every parameter perturbation."""
    if N == 1:
        return np.zeros((1, 1))
    r = np.random.default_rng(seed)
    k = max(1, int(round(0.2 * N)))
    W = np.zeros((N, N))
    for i in range(N):
        tgt = r.choice([j for j in range(N) if j != i], size=min(k, N - 1),
                       replace=False)
        W[i, tgt] = 1.0
    return W


def rhs(t, y, g, W, N):
    _simguard.check()
    V = y[0:N]
    m = y[N:2 * N]
    h = y[2 * N:3 * N]
    n = y[3 * N:4 * N]
    gNa, gK, gL = g[0::3], g[1::3], g[2::3]

    d1 = V + 40.0
    a_m = np.where(np.abs(d1) < 1e-6, 1.0, 0.1 * d1 / (1 - np.exp(-d1 / 10.0)))
    b_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
    a_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    b_h = 1.0 / (1 + np.exp(-(V + 35.0) / 10.0))
    d2 = V + 55.0
    a_n = np.where(np.abs(d2) < 1e-6, 0.1, 0.01 * d2 / (1 - np.exp(-d2 / 10.0)))
    b_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

    s = 1.0 / (1.0 + np.exp(-(V - V_HALF) / V_SLOPE))       # presynaptic gate
    I_syn = G_SYN * (W.T @ s) * (V - E_SYN)

    dV = (I_DRIVE - gNa * m**3 * h * (V - E_Na) - gK * n**4 * (V - E_K)
          - gL * (V - E_L) - I_syn) / C_m
    return np.concatenate([dV,
                           a_m * (1 - m) - b_m * m,
                           a_h * (1 - h) - b_h * h,
                           a_n * (1 - n) - b_n * n])


def simulate(g, N, W):
    r = np.random.default_rng(1234)
    y0 = np.concatenate([-65.0 + r.uniform(-5, 5, N), np.full(N, 0.05),
                         np.full(N, 0.6), np.full(N, 0.32)])
    te = np.arange(0, T_END, DT)
    _simguard.start(60.0)
    try:
        sol = solve_ivp(rhs, (0, T_END), y0, args=(g, W, N), t_eval=te,
                        method="RK45", rtol=1e-6, atol=1e-8, max_step=0.1)
    except _simguard.Budget:
        return None
    if not sol.success or sol.y.shape[1] < len(te):
        return None
    return sol.y[0:N]                                        # voltage of every cell


def observe(g, N, W, stride):
    V = simulate(g, N, W)
    if V is None or not np.all(np.isfinite(V)):
        return None
    return V[:, ::stride].ravel().astype(float)


def _col(arg):
    """One parameter perturbed both ways; returns the whitened difference column."""
    N, k, stride, seed = arg
    W = wiring(N, seed)
    g = np.tile(G_STAR, N)
    gp, gm = g.copy(), g.copy()
    gp[k] *= np.exp(DELTA)
    gm[k] *= np.exp(-DELTA)
    a = observe(gp, N, W, stride)
    b = observe(gm, N, W, stride)
    if a is None or b is None:
        return k, None
    return k, ((a - b) / (2 * DELTA)).astype(np.float64)


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
    out = {"question": "does the constrained fraction depend on network size when "
                       "the cell model, the parameter type, the wiring and the "
                       "observable budget are all held fixed?",
           "cell_model": "Hodgkin-Huxley, three conductances per cell",
           "fixed": {"synaptic_strength": G_SYN, "drive_uA": I_DRIVE,
                     "delta_log": DELTA, "T_ms": T_END},
           "rows": []}

    for N in SIZES:
        npar = 3 * N
        W = wiring(N)
        g = np.tile(G_STAR, N)
        base = simulate(g, N, W)
        if base is None:
            print(f"  N={N}: baseline failed", flush=True)
            continue
        n_t = base.shape[1]
        rates = [int(((base[i, :-1] < 0) & (base[i, 1:] >= 0)).sum()) for i in range(N)]

        for ratio in OBS_PER_PAR:
            want = ratio * npar
            stride = max(1, int(np.ceil(N * n_t / want)))
            jobs = [(N, k, stride, 0) for k in range(npar)]
            cols = {}
            with ProcessPoolExecutor(max_workers=min(NW, npar),
                                     mp_context=CTX) as pool:
                for k, v in pool.map(_col, jobs, chunksize=1):
                    cols[k] = v
            good = [k for k in range(npar) if cols[k] is not None]
            if len(good) < npar:
                print(f"  N={N} ratio={ratio}: {npar-len(good)} columns failed",
                      flush=True)
            if not good:
                continue
            L = min(len(cols[k]) for k in good)
            J = np.array([cols[k][:L] for k in good]).T
            sd = J.std(axis=1)
            keep = sd > 1e-12 * max(sd.max(), 1e-30)
            Jw = J[keep] / sd[keep][:, None] if keep.any() else J
            sp = spectrum(Jw.T @ Jw)
            if sp is None:
                continue
            row = {"n_cells": N, "n_params": npar, "n_obs": int(Jw.shape[0]),
                   "obs_per_param_target": ratio,
                   "obs_per_param_actual": round(Jw.shape[0] / npar, 2),
                   "n_dead_params": int(npar - len(good)),
                   "mean_spikes_per_cell": float(np.mean(rates)), **sp,
                   "constrained_fraction_90": sp["eff_dim_90"] / npar,
                   "constrained_fraction_99": sp["eff_dim_99"] / npar}
            out["rows"].append(row)
            print(f"  N={N:3d}  par {npar:4d}  obs {row['n_obs']:6d} "
                  f"({row['obs_per_param_actual']:5.1f}x)  "
                  f"eff {sp['eff_dim_90']:3d}/{sp['eff_dim_99']:3d}  "
                  f"frac90 {row['constrained_fraction_90']:.3f}  "
                  f"PR {sp['participation_ratio']:.2f}  "
                  f"{sp['spectrum_decades']:.1f}dec  "
                  f"spk/cell {np.mean(rates):.1f}  [{time.time()-t0:.0f}s]",
                  flush=True)

    out["elapsed_sec"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nsaved -> {OUT}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
