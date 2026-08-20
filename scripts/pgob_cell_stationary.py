"""Choose how much of each trace to discard, instead of assuming 40% everywhere.

The A-current neuron falls silent partway through its window and resumes, so a
fixed 40% discard begins reading in the middle of the silent stretch: the
amplitude quantiles, spectrum and autocorrelation are then computed across a
discontinuity, and one seed in twenty-four matched the behaviour. Lengthening
the window removes the silence but flattens the basin -- every model converged
worse at four times the window -- so the fix has to be the discard fraction, not
the window.

The fraction is therefore chosen per model by stationarity: keep the shortest
tail whose quarters have the same range, so that whatever is retained is the
settled oscillation and nothing else. Applied uniformly, it changes nothing for
the models that were already stationary at 40% and repairs the one that was not.
"""
import json
import numpy as np
import pgob_cell_panel as P

DROPS = [0.4, 0.5, 0.6, 0.7]
TOL = 0.15                    # quarters must agree to within 15% of their mean


def stationarity(x, drop):
    seg = x[int(drop * len(x)):]
    if len(seg) < 80:
        return None
    q = np.array([np.ptp(seg[i * len(seg) // 4:(i + 1) * len(seg) // 4])
                  for i in range(4)])
    m = q.mean()
    if m <= 0:
        return None
    return float(q.std() / m), float(m)


def choose(name):
    f, y0, star, names, oi, T, cls = P.PANEL[name]
    from scipy.integrate import solve_ivp
    te = np.linspace(0, T, 4000)
    sol = solve_ivp(f, (0, T), y0, args=(np.array(star, float),), t_eval=te,
                    method="LSODA", rtol=1e-6, atol=1e-9)
    if not sol.success:
        return None
    x = sol.y[oi]
    rows = []
    for d in DROPS:
        s = stationarity(x, d)
        if s is None:
            continue
        rows.append((d, s[0], s[1]))
    if not rows:
        return None
    ok = [r for r in rows if r[1] <= TOL]
    best = ok[0] if ok else min(rows, key=lambda r: r[1])
    return {"chosen_drop": best[0], "cv_at_choice": best[1],
            "range_at_choice": best[2], "stationary": best[1] <= TOL,
            "all": {str(r[0]): round(r[1], 4) for r in rows}}


if __name__ == "__main__":
    out = {}
    print(f"quarter-range coefficient of variation by discard fraction "
          f"(stationary at <= {TOL})")
    for n in P.PANEL:
        r = choose(n)
        if r is None:
            print(f"  {n:17s} baseline unusable")
            continue
        out[n] = r
        cells = "  ".join(f"{k}:{v:.3f}" for k, v in r["all"].items())
        flag = "" if r["stationary"] else "   <-- never settles"
        mark = "*" if r["chosen_drop"] != 0.4 else " "
        print(f"  {n:17s} {cells}   chosen {r['chosen_drop']}{mark}{flag}")
    json.dump(out, open("pgob_cell_stationary.json", "w"), indent=1)
    changed = [n for n, r in out.items() if r["chosen_drop"] != 0.4]
    print(f"\n{len(changed)} model(s) need a different discard: {changed}")
