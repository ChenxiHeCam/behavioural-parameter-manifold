"""Paper2 cross-species (MAMMAL): Virtual Rodent (dm_control, 38 actuators) behavioural manifold.
Parallel to FlyGym-48d but mammal. Per actuator-gain Hessian -> eff-dim (identifiable dimension of
the 38-actuator system); and multi-behaviour: does the union stiff subspace grow / saturate as you
switch motor patterns ('actions')? Tests the cross-species 'identifiability = behaviour richness' law.
CAVEAT: open-loop sinusoidal drives (deterministic motor patterns), NOT trained natural behaviours
(those need the Aldarondo policy). This measures the actuator->behaviour identifiable manifold."""
import numpy as np, json, time, os
os.environ['MUJOCO_GL'] = 'osmesa'
from dm_control import mjcf
from dm_control.locomotion.walkers import rodent
from dm_control.locomotion.arenas import floors

arena = floors.Floor(); walker = rodent.Rat(); arena.add_free_entity(walker)
physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
NU = physics.model.nu                      # 38 actuators
root = walker.root_body
DT = physics.timestep()
NSTEP = 200; SETTLE = 20; DELTA = 0.25
rng = np.random.default_rng(0)
OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'height_std', 'heading_rms']


def make_pattern(freq, phases, amp):
    t = np.arange(NSTEP)
    return amp * np.sin(2 * np.pi * freq * t[:, None] * DT + phases[None, :])   # (NSTEP, NU)


def obs_from_com(coms):
    coms = np.asarray(coms); xy = coms[:, :2]; z = coms[:, 2]
    d = np.diff(xy, axis=0); speed = np.linalg.norm(d, axis=1) / DT
    head = np.arctan2(d[:, 1], d[:, 0]); dh = np.arctan2(np.sin(np.diff(head)), np.cos(np.diff(head)))
    return {'net_disp': float(np.linalg.norm(xy[-1] - xy[0])), 'path_len': float(speed.sum() * DT),
            'mean_speed': float(speed.mean()), 'speed_std': float(speed.std()),
            'height_std': float(z.std()), 'heading_rms': float(np.sqrt((dh ** 2).mean())) if dh.size else 0.0}


def run(pattern, gain_mult):
    physics.reset()
    for _ in range(SETTLE):
        physics.step()
    coms = []
    for t in range(NSTEP):
        physics.data.ctrl[:] = np.clip(pattern[t] * gain_mult, -1, 1)
        physics.step(); coms.append(physics.bind(root).xpos.copy())
    return obs_from_com(coms)


def eff_dim(w):
    w = np.maximum(w, 0); w = w[w > 0]
    cs = np.cumsum(np.sort(w)[::-1]) / w.sum()
    return int(np.searchsorted(cs, 0.9) + 1), int(np.searchsorted(cs, 0.99) + 1)


# define BEHAVIOURS = distinct motor patterns (different gaits/actions)
BEHAVIOURS = {}
for i, (f, a) in enumerate([(2.0, 0.4), (3.0, 0.5), (4.0, 0.5), (1.5, 0.6), (3.0, 0.3), (5.0, 0.4)]):
    BEHAVIOURS[f'pat{i}_f{f}'] = make_pattern(f, rng.uniform(0, 2 * np.pi, NU), a)

t0 = time.time()
one = np.ones(NU)
beh_res = {}
Js = []
for bn, pat in BEHAVIOURS.items():
    base = run(pat, one)
    bvec = np.array([base[k] for k in OBS]); scale = np.abs(bvec) + 1e-9
    J = np.zeros((len(OBS), NU))
    for j in range(NU):
        gp = one.copy(); gp[j] = 1 + DELTA; gm = one.copy(); gm[j] = 1 - DELTA
        op = run(pat, gp); om = run(pat, gm)
        J[:, j] = (np.array([op[k] for k in OBS]) - np.array([om[k] for k in OBS])) / scale / (2 * DELTA)
    H = J.T @ J
    e90, e99 = eff_dim(np.linalg.eigvalsh(H))
    beh_res[bn] = {'eff_dim_90': e90, 'eff_dim_99': e99, 'net_disp': base['net_disp']}
    Js.append(J)
    print(f'{bn}: eff-dim {e90}/{e99} of {NU}  ({time.time()-t0:.0f}s)', flush=True)

# saturation: union eff-dim vs number of behaviours
rng2 = np.random.default_rng(1); nB = len(Js)
import math
curve = []
for k in range(1, nB + 1):
    es = []
    seen = set()
    for _ in range(40):
        if len(seen) >= min(15, math.comb(nB, k)):
            break
        sub = tuple(sorted(rng2.choice(nB, k, replace=False)))
        if sub in seen:
            continue
        seen.add(sub)
        Jc = np.vstack([Js[i] for i in sub])
        es.append(eff_dim(np.linalg.eigvalsh(Jc.T @ Jc))[1])
    curve.append({'n_behaviours': k, 'eff_dim_99_mean': float(np.mean(es))})
Jall = np.vstack(Js); wall = np.sort(np.maximum(np.linalg.eigvalsh(Jall.T @ Jall), 0))[::-1]
u90, u99 = eff_dim(wall)

out = {
    'experiment': 'cross-species MAMMAL: Virtual Rodent (dm_control, 38 actuators) manifold',
    'n_actuators': NU, 'caveat': 'open-loop sinusoidal drives, not trained natural behaviours',
    'per_behaviour': beh_res,
    'saturation_curve': curve,
    'union_eff_dim_90_99': [u90, u99],
    'union_spectrum_top10': [round(float(x), 4) for x in wall[:10]],
    'elapsed_sec': round(time.time() - t0, 1),
}
json.dump(out, open('/root/autodl-tmp/paper2_RODENT_manifold.json', 'w'), indent=2)
print('=== Virtual Rodent (mammal) manifold ===', flush=True)
print('per-behaviour eff-dim:', {b: f"{v['eff_dim_90']}/{v['eff_dim_99']}" for b, v in beh_res.items()}, flush=True)
print('saturation:', [(c['n_behaviours'], round(c['eff_dim_99_mean'], 2)) for c in curve], flush=True)
print(f'UNION eff-dim {u90}/{u99} of {NU} | top10 spectrum {[round(float(x),3) for x in wall[:10]]}', flush=True)
print('RODENT_DONE %.0fs' % (time.time() - t0), flush=True)
