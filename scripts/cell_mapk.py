# -*- coding: utf-8 -*-
"""PGOB on an A-class mechanistic cell model: Kholodenko (2000) MAPK cascade.
A negative-feedback signalling ODE (8 species) that sustains oscillations of
doubly-phosphorylated MAPK. This is the canonical 'sloppy' systems-biology model
(the family Gutenkunst/Sethna analysed), so it is the natural cell-scale target
for PGOB's degeneracy-embracing, observable-only calibration.

Step 1 here: encode the forward model f_theta, verify sustained oscillation, and
define the 'behavioural observables' of the cell (period, amplitude, mean, etc.).
"""
import numpy as np
from scipy.integrate import solve_ivp
import _simguard

# --- canonical Kholodenko 2000 parameters (theta*) ---
# conserved totals
MKKK_T, MKK_T, MAPK_T = 100.0, 300.0, 300.0
THETA_NAMES = ["V1","Ki","V2","k3","k4","V5","V6","k7","k8","V9","V10"]
THETA_STAR = np.array([2.5, 9.0, 0.25, 0.025, 0.025, 0.75, 0.75, 0.025, 0.025, 0.5, 0.5])
# Michaelis constants held fixed (structural, not searched here)
K1=10.0; KK2=8.0; KK3=15.0; KK4=15.0; KK5=15.0; KK6=15.0; KK7=15.0; KK8=15.0; KK9=15.0; KK10=15.0
n_hill = 1.0

def rhs(t, y, th):
    _simguard.check()
    V1,Ki,V2,k3,k4,V5,V6,k7,k8,V9,V10 = th
    MKKK, MKKKp, MKK, MKKp, MKKpp, MAPK, MAPKp, MAPKpp = y
    v1 = V1*MKKK / ((1.0 + (MAPKpp/Ki)**n_hill) * (K1 + MKKK))
    v2 = V2*MKKKp / (KK2 + MKKKp)
    v3 = k3*MKKKp*MKK / (KK3 + MKK)
    v4 = k4*MKKKp*MKKp / (KK4 + MKKp)
    v5 = V5*MKKpp / (KK5 + MKKpp)
    v6 = V6*MKKp / (KK6 + MKKp)
    v7 = k7*MKKpp*MAPK / (KK7 + MAPK)
    v8 = k8*MKKpp*MAPKp / (KK8 + MAPKp)
    v9 = V9*MAPKpp / (KK9 + MAPKpp)
    v10 = V10*MAPKp / (KK10 + MAPKp)
    return [v2-v1, v1-v2, v6-v3, v3+v5-v4-v6, v4-v5, v10-v7, v7+v9-v8-v10, v8-v9]

def simulate(th, t_end=6000.0, dt=10.0):
    y0 = [MKKK_T*0.9, MKKK_T*0.1, MKK_T*0.8, MKK_T*0.15, MKK_T*0.05,
          MAPK_T*0.8, MAPK_T*0.15, MAPK_T*0.05]
    t_eval = np.arange(0, t_end, dt)
    _simguard.start(3.0)
    try:
        sol = solve_ivp(rhs, (0, t_end), y0, args=(th,), t_eval=t_eval,
                        method="LSODA", rtol=1e-5, atol=1e-7)
    except _simguard.Budget:
        return t_eval, np.full_like(t_eval, np.nan)
    if not sol.success or sol.y.shape[1] < len(t_eval):
        return t_eval, np.full_like(t_eval, np.nan)
    return sol.t, sol.y[7]  # MAPK-PP trace = the observable signal

def observables(th):
    """The 'behavioural observables' of the cell: features of the MAPK-PP signal
    over the last 60% of the trace (steady oscillatory regime)."""
    t, x = simulate(th)
    if not np.all(np.isfinite(x)):
        return None
    m = len(x); seg = x[int(0.4*m):]; tseg = t[int(0.4*m):]
    mean = seg.mean(); amp = seg.max() - seg.min()
    # dominant oscillation period via zero-crossings of the de-meaned signal
    s = seg - mean; zc = np.where(np.diff(np.sign(s)) > 0)[0]
    if len(zc) >= 2:
        period = float(np.mean(np.diff(tseg[zc])))
    else:
        period = 0.0  # non-oscillatory
    return np.array([mean, amp, period, seg.max(), seg.min()])

if __name__ == "__main__":
    obs = observables(THETA_STAR)
    print("theta* observables [mean, amp, period, max, min]:")
    print("  ", np.round(obs, 3))
    print("oscillates:", obs[2] > 0, "| period ~", round(obs[2], 1), "s")
