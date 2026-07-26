"""cross-species (Drosophila LARVA): larvaworld behavioural manifold.
Fills the cross-species gradient between worm (saturates ~4) and rodent (grows to 12+).
Perturb ~10 behavioural-module parameters (crawler/turner/interference/intermitter) and measure
the trajectory Hessian -> eff-dim, across DISTINCT behaviours (explore/chemotaxis/wind/feeding =
genuinely different sensory contexts). Does the union manifold grow with behavioural richness?"""
import numpy as np, json, time, sys, io, contextlib, copy
from larvaworld.lib import reg, sim

DELTA = 0.25; DUR = 0.5
# behavioural-module params to perturb (module, param)
PARAMS = [('crawler', 'amp'), ('crawler', 'freq'),
          ('turner', 'base_activation'), ('turner', 'w_cc'), ('turner', 'w_ce'),
          ('turner', 'w_ec'), ('turner', 'w_ee'), ('turner', 'tau'),
          ('interference', 'attenuation'), ('intermitter', 'crawl_freq')]
BEHAVIOURS = ['dish', 'chemorbit', 'anemotaxis', 'RvsS_on']
OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'bend_rms', 'heading_rms']


def defaults():
    m = reg.conf.Model.getID('explorer')['brain']
    d = {}
    for mod, p in PARAMS:
        v = m.get(mod, {}).get(p) if m.get(mod) else None
        if isinstance(v, (int, float)):
            d[(mod, p)] = float(v)
    return d


DEF = defaults()
PARAMS = [k for k in PARAMS if k in DEF]
print('perturbable params:', [f'{a}.{b}' for a, b in PARAMS], flush=True)


def run(expid, mults):
    e = reg.conf.Exp.getID(expid)        # reference (keep proper object; don't deepcopy whole config)
    grp = e['larva_groups']; k = list(grp.keys())[0]
    orig = grp[k]['model']               # save to restore (avoid permanent cache mutation)
    try:
        mod = copy.deepcopy(reg.conf.Model.getID(orig)) if isinstance(orig, str) else copy.deepcopy(orig)
        for (module, p), mult in mults.items():
            mod['brain'][module][p] = DEF[(module, p)] * mult
        grp[k]['model'] = mod
        with contextlib.redirect_stdout(io.StringIO()):
            r = sim.ExpRun(parameters=e, duration=DUR, N=1, store_data=False)
            r.simulate()
        s = r.datasets[0].s
        x = np.asarray(s['x'].values, float); y = np.asarray(s['y'].values, float)
        bend = np.asarray(s['bend'].values, float) if 'bend' in s else np.zeros_like(x)
        ok = np.isfinite(x) & np.isfinite(y); x, y, bend = x[ok], y[ok], bend[ok]
        if len(x) < 5:
            return None
        dx = np.diff(x); dy = np.diff(y); speed = np.sqrt(dx * dx + dy * dy)
        head = np.arctan2(dy, dx); dh = np.arctan2(np.sin(np.diff(head)), np.cos(np.diff(head)))
        return {'net_disp': float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)),
                'path_len': float(speed.sum()), 'mean_speed': float(speed.mean()), 'speed_std': float(speed.std()),
                'bend_rms': float(np.sqrt(np.nanmean(bend ** 2))) if bend.size else 0.0,
                'heading_rms': float(np.sqrt((dh ** 2).mean())) if dh.size else 0.0}
    except Exception as ex:
        print(f'  run err {expid}: {repr(ex)[:80]}', flush=True)
        return None
    finally:
        grp[k]['model'] = orig          # restore cached config


def eff_dim(w):
    w = np.maximum(w, 0); w = w[w > 0]
    if w.sum() <= 0:
        return 0, 0
    cs = np.cumsum(np.sort(w)[::-1]) / w.sum()
    return int(np.searchsorted(cs, 0.9) + 1), int(np.searchsorted(cs, 0.99) + 1)


t0 = time.time()
one = {k: 1.0 for k in PARAMS}
beh = {}; Js = []
for bn in BEHAVIOURS:
    base = run(bn, one)
    if base is None:
        print(f'{bn}: base FAILED, skip', flush=True); continue
    bvec = np.array([base[k] for k in OBS]); scale = np.abs(bvec) + 1e-9
    J = np.zeros((len(OBS), len(PARAMS))); ok = True
    for j, pk in enumerate(PARAMS):
        mp = dict(one); mp[pk] = 1 + DELTA; mm = dict(one); mm[pk] = 1 - DELTA
        op = run(bn, mp); om = run(bn, mm)
        if op is None or om is None:
            ok = False; break
        J[:, j] = (np.array([op[k] for k in OBS]) - np.array([om[k] for k in OBS])) / scale / (2 * DELTA)
    if not ok:
        print(f'{bn}: a perturbation failed, skip', flush=True); continue
    e90, e99 = eff_dim(np.linalg.eigvalsh(J.T @ J))
    beh[bn] = {'eff_dim_90': e90, 'eff_dim_99': e99, 'net_disp': base['net_disp']}
    Js.append(J)
    print(f'{bn}: eff-dim {e90}/{e99} of {len(PARAMS)}  ({time.time()-t0:.0f}s)', flush=True)

nB = len(Js); union90 = union99 = 0; spectrum = []
if nB:
    Jall = np.vstack(Js); wall = np.sort(np.maximum(np.linalg.eigvalsh(Jall.T @ Jall), 0))[::-1]
    union90, union99 = eff_dim(wall); spectrum = [round(float(x), 4) for x in wall]

out = {'experiment': 'cross-species LARVA: larvaworld behavioural manifold',
       'n_params': len(PARAMS), 'params': [f'{a}.{b}' for a, b in PARAMS],
       'behaviours_used': list(beh.keys()), 'per_behaviour': beh,
       'union_eff_dim_90_99': [union90, union99], 'union_spectrum': spectrum,
       'elapsed_sec': round(time.time() - t0, 1)}
json.dump(out, open('/root/autodl-tmp/LARVA_manifold.json', 'w'), indent=2)
print('=== Drosophila LARVA manifold ===', flush=True)
print('per-behaviour:', {b: f"{v['eff_dim_90']}/{v['eff_dim_99']}" for b, v in beh.items()}, flush=True)
print(f'UNION eff-dim {union90}/{union99} of {len(PARAMS)} ({nB} behaviours)', flush=True)
print('LARVA_DONE %.0fs' % (time.time() - t0), flush=True)
