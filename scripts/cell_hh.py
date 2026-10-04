# -*- coding: utf-8 -*-
"""Second A-class cell model, close to our own conductance work: the
Hodgkin-Huxley (1952) conductance-based neuron. Parameters theta = ionic
conductances (g_Na, g_K, g_L) -- structurally identical to the ion-channel
conductances PGOB already calibrates in BAAIWorm, but here for a single CELL.
Observable = the neuron's firing 'behaviour': an F-I curve (firing rate vs
injected current) plus spike-shape features. Real patch-clamp spike data are
abundant (e.g. Allen Cell Types Database), so this model also supports PGOB-real.
"""
import numpy as np
from scipy.integrate import solve_ivp
import _simguard

# reversal potentials / capacitance (fixed, structural)
E_Na, E_K, E_L, C_m = 50.0, -77.0, -54.4, 1.0
THETA_NAMES = ["g_Na", "g_K", "g_L"]
THETA_STAR = np.array([120.0, 36.0, 0.3])   # canonical HH conductances (mS/cm^2)
I_LEVELS = np.array([4.0, 7.0, 10.0, 15.0, 20.0])  # injected currents (uA/cm^2)

def _a_m(V):
    d = V+40.0
    return np.where(np.abs(d)<1e-6, 1.0, 0.1*d/(1-np.exp(-d/10.0)))
def _b_m(V): return 4.0*np.exp(-(V+65.0)/18.0)
def _a_h(V): return 0.07*np.exp(-(V+65.0)/20.0)
def _b_h(V): return 1.0/(1+np.exp(-(V+35.0)/10.0))
def _a_n(V):
    d = V+55.0
    return np.where(np.abs(d)<1e-6, 0.1, 0.01*d/(1-np.exp(-d/10.0)))
def _b_n(V): return 0.125*np.exp(-(V+65.0)/80.0)

def rhs(t, y, th, I):
    _simguard.check()
    gNa, gK, gL = th
    V, m, h, n = y
    dV = (I - gNa*m**3*h*(V-E_Na) - gK*n**4*(V-E_K) - gL*(V-E_L))/C_m
    return [dV, _a_m(V)*(1-m)-_b_m(V)*m, _a_h(V)*(1-h)-_b_h(V)*h, _a_n(V)*(1-n)-_b_n(V)*n]

def _diverged(t, y, th, I):
    return 500.0 - abs(y[0])
_diverged.terminal = True
_diverged.direction = -1


def _trace(th, I, T=80.0, dt=0.02):
    y0 = [-65.0, 0.05, 0.6, 0.32]
    te = np.arange(0, T, dt)
    try:
        sol = solve_ivp(rhs, (0, T), y0, args=(th, I), t_eval=te, method="RK45",
                        rtol=1e-5, atol=1e-7, max_step=0.1, events=_diverged)
    except _simguard.Budget:
        return te, np.full(len(te), np.nan)
    if sol.y.shape[1] < len(te):          # terminated early, or diverged
        return te, np.full(len(te), np.nan)
    return sol.t, sol.y[0]

def _spike_count(t, V, thr=0.0):
    up = np.where((V[:-1] < thr) & (V[1:] >= thr))[0]
    return len(up)

def observables(th):
    """Firing 'behaviour' fingerprint: firing rate at each current level +
    spike amplitude + AP half-width at a reference current.

    The budget covers this whole call. It previously wrapped each of the six
    traces separately, so a divergent parameter set could consume six budgets
    rather than one and a single optimisation ran for hours."""
    _simguard.start(4.0)
    try:
        rates = []
        for I in I_LEVELS:
            t, V = _trace(th, I)
            if not np.all(np.isfinite(V)): return None
            rates.append(_spike_count(t, V) / (t[-1]/1000.0))   # Hz
        # spike shape at reference current
        t, V = _trace(th, 15.0)
        amp = float(V.max() - V.min())
        above = V > (V.min() + 0.5*amp)
        halfwidth = float(above.sum() * (t[1]-t[0]))
        return np.array(rates + [amp, halfwidth])
    except Exception:
        return None

if __name__ == "__main__":
    ob = observables(THETA_STAR)
    print("theta* (g_Na,g_K,g_L) =", THETA_STAR)
    print("observable [rate@4,7,10,15,20 Hz, spike_amp mV, AP_halfwidth ms]:")
    print("  ", np.round(ob, 2))
    print("fires:", (ob[:5] > 0).sum(), "/5 current levels")
