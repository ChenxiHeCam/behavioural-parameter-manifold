"""Noise floor of the larvaworld probe.

The larva Jacobian was assembled from single unseeded realisations, and one of
the ten perturbed parameters drives the intermitter, the module that generates
stochastic crawl and pause bouts. Whether the measured columns carry parameter
sensitivity or run-to-run variation has never been checked, and the same failure
was diagnosed in FlyGym, where three quarters of the apparent difference at the
original step was controller stochasticity.

This repeats the baseline at identical parameters and compares the spread of the
six observables against the differences the perturbations produced.
"""
import io, contextlib, copy, json, time
import numpy as np
from larvaworld.lib import reg, sim

DELTA = 0.25
DUR = 0.5
N_REPEAT = 12
PARAMS = [('crawler', 'amp'), ('crawler', 'freq'),
          ('turner', 'base_activation'), ('turner', 'w_cc'), ('turner', 'w_ce'),
          ('turner', 'w_ec'), ('turner', 'w_ee'), ('turner', 'tau'),
          ('interference', 'attenuation'), ('intermitter', 'crawl_freq')]
OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'bend_rms', 'heading_rms']
BEH = 'dish'


def defaults():
    m = reg.conf.Model.getID('explorer')['brain']
    d = {}
    for mod, p in PARAMS:
        v = m.get(mod, {}).get(p) if m.get(mod) else None
        if isinstance(v, (int, float)):
            d[(mod, p)] = float(v)
    return d


DEF = defaults()
PAR = [k for k in PARAMS if k in DEF]


def run(expid, mults):
    e = reg.conf.Exp.getID(expid)
    grp = e['larva_groups']; k = list(grp.keys())[0]
    orig = grp[k]['model']
    try:
        mod = copy.deepcopy(reg.conf.Model.getID(orig)) if isinstance(orig, str) else copy.deepcopy(orig)
        for (module, p), mult in mults.items():
            mod['brain'][module][p] = DEF[(module, p)] * mult
        grp[k]['model'] = mod
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = sim.ExpRun(parameters=e, duration=DUR, N=1, store_data=False)
            r.simulate()
        s = r.datasets[0].s
        x = np.asarray(s['x'].values, float); y = np.asarray(s['y'].values, float)
        bend = np.asarray(s['bend'].values, float) if 'bend' in s else np.zeros_like(x)
        ok = np.isfinite(x) & np.isfinite(y); x, y, bend = x[ok], y[ok], bend[ok]
        if len(x) < 5:
            return None
        dx = np.diff(x); dy = np.diff(y); sp = np.sqrt(dx * dx + dy * dy)
        head = np.arctan2(dy, dx); dh = np.arctan2(np.sin(np.diff(head)), np.cos(np.diff(head)))
        return np.array([np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2), sp.sum(),
                         sp.mean(), sp.std(),
                         np.sqrt(np.nanmean(bend**2)) if bend.size else 0.0,
                         np.sqrt((dh**2).mean()) if dh.size else 0.0])
    except Exception as ex:
        print("  run err", repr(ex)[:90], flush=True)
        return None
    finally:
        grp[k]['model'] = orig


t0 = time.time()
one = {k: 1.0 for k in PAR}
print(f"repeating the baseline {N_REPEAT} times at identical parameters", flush=True)
reps = []
for i in range(N_REPEAT):
    v = run(BEH, one)
    if v is not None:
        reps.append(v)
    print(f"  {i+1}/{N_REPEAT}  {time.time()-t0:.0f}s", flush=True)
R = np.array(reps)
base = R.mean(0)
scale = np.abs(base) + 1e-9
noise = (R / scale).std(0)          # whitened run-to-run spread
print("\nwhitened run-to-run spread per observable:")
for k, nm in enumerate(OBS):
    print(f"  {nm:12s} mean {base[k]:10.4f}   rel sd {noise[k]:.4f}")

print(f"\nperturbation-induced whitened differences, +-{DELTA}", flush=True)
rows = []
for (mod, p) in PAR:
    mp = dict(one); mp[(mod, p)] = 1 + DELTA
    mm = dict(one); mm[(mod, p)] = 1 - DELTA
    a, b = run(BEH, mp), run(BEH, mm)
    if a is None or b is None:
        print(f"  {mod}.{p}: failed"); continue
    d = np.abs(a - b) / scale
    rows.append((f"{mod}.{p}", d))
    snr = d / np.maximum(noise, 1e-12)
    print(f"  {mod}.{p:18s} |diff| {d.mean():.4f}   SNR vs noise {snr.mean():6.2f}", flush=True)

allsnr = np.array([ (d / np.maximum(noise, 1e-12)).mean() for _, d in rows])
print(f"\nmedian SNR across the ten parameters: {np.median(allsnr):.2f}")
print(f"parameters whose signal exceeds the noise floor: "
      f"{(allsnr > 1).sum()}/{len(allsnr)}")
json.dump({"n_repeat": int(len(R)), "observables": OBS,
           "baseline_mean": base.tolist(),
           "whitened_run_to_run_sd": noise.tolist(),
           "per_parameter_mean_abs_diff": {n: d.mean() for n, d in rows},
           "per_parameter_snr": {n: float((d/np.maximum(noise,1e-12)).mean()) for n, d in rows},
           "median_snr": float(np.median(allsnr)),
           "n_params_above_noise": int((allsnr > 1).sum()),
           "elapsed_sec": round(time.time()-t0,1)},
          open("D:/Warm/LARVA_noise_floor.json","w"), indent=1)
print("saved -> LARVA_noise_floor.json")
