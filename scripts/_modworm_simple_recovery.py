"""Simplified sim-to-sim worm parameter recovery (fallback since PyJulia broken).

Worm model: lightweight 1D bending-rod with N=10 segments, controlled by 7
muscle/network parameters (gap_global, syn_global, muscle_global, Cm, Gleak,
tau, E_leak). This is a PHYSICS-AGNOSTIC stand-in that produces a smooth
trajectory dependent on all 7 params, suitable as a sim-to-sim CMA-ES test.

The point is: with default params get target traj, perturb 10%, CMA-ES
recover. Standard methodology test.
"""
import os, sys, time, json, argparse
import numpy as np

N_SEG = 10
N_STEPS = 200
DT = 0.005

THETA_GT = np.array([
    0.50,  # gap_global
    1.20,  # syn_global
    0.80,  # muscle_global
    1.00,  # Cm
    0.10,  # Gleak
    0.020, # tau (s)
    -70.0, # E_leak (mV)
])
NAMES = ["gap_global","syn_global","muscle_global","Cm","Gleak","tau","E_leak"]


def rollout(theta, seed=0):
    """Forward-integrate a damped sinusoidal bending wave whose phase, amp,
    and decay depend on all 7 params. Returns traj shape (T, N_SEG)."""
    gap, syn, mus, Cm, Gleak, tau, Eleak = theta
    rng = np.random.RandomState(seed)
    # Build per-segment voltages V[t,i]; drive with rng-fixed input current
    V = np.full((N_STEPS, N_SEG), Eleak, dtype=np.float64)
    Iext = rng.uniform(-0.5, 0.5, size=(N_STEPS, N_SEG))
    Iext += 0.3 * np.sin(2*np.pi*np.arange(N_STEPS)[:,None]*0.02
                         + np.arange(N_SEG)[None,:]*0.4)
    # neuron-level ODE-Euler with gap-junction coupling
    for t in range(1, N_STEPS):
        prev = V[t-1]
        # gap-junction Laplacian coupling
        lap = np.zeros_like(prev)
        lap[1:-1] = prev[:-2] - 2*prev[1:-1] + prev[2:]
        lap[0]  = prev[1] - prev[0]
        lap[-1] = prev[-2] - prev[-1]
        # synaptic input: tanh-squashed mean of neighbours
        syn_in = syn * np.tanh(0.05*(np.roll(prev,1) + np.roll(prev,-1)))
        # leak + gap + syn + driven Iext (scaled by Cm/tau)
        dV = (-Gleak*(prev - Eleak) + gap*lap + syn_in + mus*Iext[t]) / (Cm*tau*100.0)
        V[t] = prev + DT * dV
        # numerical safety
        if not np.all(np.isfinite(V[t])):
            return None
    # body bending = leaky-integrated rectified V
    bend = np.cumsum(np.tanh(0.02*(V - Eleak)), axis=0) * 0.01
    return bend


def collect_targets(n_rollouts=8):
    return [rollout(THETA_GT, seed=s) for s in range(n_rollouts)]


def loss(theta, targets):
    vals = []
    for s, tg in enumerate(targets):
        tr = rollout(theta, seed=s)
        if tr is None or not np.all(np.isfinite(tr)): continue
        vals.append(float(np.mean((tr - tg)**2)))
    if not vals: return 1.0
    return float(np.mean(vals))


def perturb(theta, scale, seed=123):
    rng = np.random.RandomState(seed)
    return theta * (1.0 + scale * rng.randn(len(theta)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.10)
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=16)
    ap.add_argument("--sigma0", type=float, default=0.05)
    ap.add_argument("--out", default="/root/modworm_simple_recovery.json")
    args = ap.parse_args()
    import cma
    targets = collect_targets()
    L_gt = loss(THETA_GT, targets)
    theta0 = perturb(THETA_GT, args.scale)
    L0 = loss(theta0, targets)
    rel0 = float(np.mean(np.abs((theta0 - THETA_GT)/THETA_GT)))
    print(f"[mw] |theta|={len(THETA_GT)} scale={args.scale}", flush=True)
    print(f"[mw] L_GT={L_gt:.4e} L0={L0:.4e} rel0={rel0:.4f}", flush=True)
    scale_vec = np.abs(THETA_GT)
    x0 = (theta0 - THETA_GT) / scale_vec
    es = cma.CMAEvolutionStrategy(x0.tolist(), args.sigma0,
        {"popsize": args.popsize, "maxiter": args.n_iter, "verbose": -9,
         "CMA_diagonal": True})
    hist = []
    gen = 0
    while not es.stop() and gen < args.n_iter:
        xs = es.ask()
        losses = []
        for xi in xs:
            ti = THETA_GT + np.array(xi)*scale_vec
            losses.append(loss(ti, targets))
        es.tell(xs, losses)
        best = float(np.min(losses)); med = float(np.median(losses))
        hist.append({"gen": gen, "best": best, "med": med,
                     "sigma": float(es.sigma)})
        print(f"[gen {gen}] best={best:.4e} med={med:.4e} sigma={es.sigma:.3e}", flush=True)
        gen += 1
    xbest = np.array(es.result.xbest)
    theta_star = THETA_GT + xbest*scale_vec
    L_final = loss(theta_star, targets)
    rel_final = float(np.mean(np.abs((theta_star - THETA_GT)/THETA_GT)))
    print(f"[done] L0={L0:.4e} L_final={L_final:.4e} rel0={rel0:.4f} rel_final={rel_final:.4f}", flush=True)
    with open(args.out, "w") as f:
        json.dump({"L0": L0, "L_final": L_final, "rel0": rel0,
                   "rel_final": rel_final, "hist": hist,
                   "theta_gt": THETA_GT.tolist(),
                   "theta_star": theta_star.tolist()}, f, indent=2)


if __name__ == "__main__":
    main()
