"""Persist complete closed-loop union probes with strict rollout validation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'rms_heading', 'fwd_world']


def measure_union(labels, spec_for, delta, food_close, food_far, experiment):
    outdir = Path(os.environ.get('PAPER2_OUTPUT_DIR', 'results/revision_20261004')) / experiment
    outdir.mkdir(parents=True, exist_ok=True)
    worker = Path(os.environ.get('BAAI_BODY_WORKER', str(Path(__file__).with_name('_b6_physics_worker.py'))))
    python = os.environ.get('BAAI_PYTHON', sys.executable)
    nworkers = int(os.environ.get('NWORK', '8'))
    def run(item):
        key, spec = item
        path = outdir / (key + '.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['spec'] != spec:
                raise ValueError('Cached rollout configuration differs')
            return key, saved['result']['obs']
        prefix = ['xvfb-run', '-a'] if os.environ.get('BAAI_USE_XVFB', '0') == '1' else []
        p = subprocess.run(prefix + [python, str(worker), json.dumps(spec)],
                           capture_output=True, text=True, timeout=1800)
        for line in p.stdout.splitlines():
            if line.startswith('RESULT '):
                result = json.loads(line[7:])
                obs = result.get('obs')
                if not obs or not all(np.isfinite(obs[k]) for k in OBS):
                    raise ValueError(f'Invalid observable in {key}')
                path.write_text(json.dumps({'spec': spec, 'result': result}, indent=2))
                return key, obs
        (outdir / (key + '.log')).write_text(p.stdout + p.stderr)
        raise RuntimeError(f'Rollout failed {key}: exit {p.returncode}')
    matrices = {}
    baselines = {}
    for tag, food in [('chemo', food_close), ('loco', food_far)]:
        jobs = [(tag+'_base', spec_for(None, 1., food))]
        for label in labels:
            for sign in (1, -1):
                jobs.append((f'{tag}_{label}_{sign:+d}', spec_for(label, float(np.exp(sign*delta)), food)))
        with ThreadPoolExecutor(max_workers=nworkers) as pool:
            responses = dict(pool.map(run, jobs))
        base = np.array([responses[tag+'_base'][k] for k in OBS])
        scale = np.abs(base) + 1e-9
        J = np.column_stack([(np.array([responses[f'{tag}_{label}_+1'][k] for k in OBS]) -
                              np.array([responses[f'{tag}_{label}_-1'][k] for k in OBS])) / (2*delta*scale)
                             for label in labels])
        matrices['J_'+tag] = J
        matrices['G_'+tag] = J.T @ J
        matrices['output_scale_'+tag] = scale
        baselines[tag] = responses[tag+'_base']
        print(tag, 'responses complete', flush=True)
    matrices['J_union'] = np.vstack([matrices['J_chemo'], matrices['J_loco']])
    matrices['G_union'] = matrices['J_union'].T @ matrices['J_union']
    summary = {}
    for name in ('chemo', 'loco', 'union'):
        J = matrices['J_'+name]
        ev = np.linalg.eigvalsh(matrices['G_'+name])[::-1]
        singular_values = np.linalg.svd(J, compute_uv=False)
        rank_tol = max(J.shape) * np.finfo(np.float64).eps * singular_values[0]
        mass = np.maximum(ev, 0)
        cumulative = np.cumsum(mass)/mass.sum() if mass.sum() else mass
        summary[name] = {'eigenvalues': ev.tolist(), 'singular_values': singular_values.tolist(),
            'numerical_rank': int((singular_values > rank_tol).sum()),
            'singular_value_rank_tolerance': float(rank_tol),
            'rank_tolerance_interpretation': 'floating-point algebraic tolerance; not a simulation noise floor',
            'rank_at_relative_singular_tolerance': {str(t): int((singular_values > t*singular_values[0]).sum())
                                                   for t in (1e-4, 1e-6, 1e-8, 1e-10)},
            'eff_dim_90': int(np.searchsorted(cumulative, .90)+1) if mass.sum() else 0,
            'eff_dim_99': int(np.searchsorted(cumulative, .99)+1) if mass.sum() else 0}
    Vc = np.linalg.eigh(matrices['G_chemo'])[1][:,-3:]
    Vl = np.linalg.eigh(matrices['G_loco'])[1][:,-3:]
    np.savez_compressed(outdir/'full_matrices.npz', **matrices)
    result = {'experiment': experiment, 'labels': labels, 'observable_names': OBS,
        'delta': delta, 'parameter_coordinate': 'log multiplicative gain', 'summary': summary,
        'stiff_subspace_overlap_top3_cos': float(np.linalg.norm(Vc.T@Vl)/np.sqrt(3)),
        'baseline_observables': baselines, 'matrix_file': 'full_matrices.npz',
        'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'worker_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
        'python': sys.version, 'numpy': np.__version__}
    (outdir/'result.json').write_text(json.dumps(result, indent=2))
    return result
