"""Flyvis sensitivity with explicit model state and dimensionless coordinates.

Positive tau/strength: theta=theta0*exp(z). Signed biases: theta=theta0+s*z,
s=1 activity unit. Zero strengths: theta=s*q, q>=0, with the corresponding
default initialization strength as s. A second-order right derivative describes
each boundary coordinate. Interior spectra are primary; inclusive boundary Gram
matrices summarize a tangent cone and are not unconstrained Fisher geometries.
"""
import argparse
import hashlib
import importlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def spectrum(J):
    J = np.asarray(J, dtype=np.float64)
    singular = np.linalg.svd(J, compute_uv=False)
    ev = np.pad(singular ** 2, (0, J.shape[1] - len(singular)))
    mass = float(ev.sum())
    rank = int(np.sum(singular > np.finfo(float).eps * max(J.shape) * singular[0])) if len(singular) else 0
    return {'n_readouts': J.shape[0], 'n_params': J.shape[1], 'eigenvalues': ev.tolist(),
            'singular_values': singular.tolist(), 'numeric_rank': rank, 'total_sensitivity': mass,
            'eff_dim_90': int(np.searchsorted(np.cumsum(ev), .9 * mass) + 1) if mass else 0,
            'eff_dim_99': int(np.searchsorted(np.cumsum(ev), .99 * mass) + 1) if mass else 0,
            'participation_ratio': float(mass ** 2 / (ev @ ev)) if mass else 0.}


def chart_entry(name, index, baseline, initialization, bias_scale):
    value = float(baseline)
    common = {'group': name, 'index': int(index), 'baseline': value}
    if name == 'nodes_bias':
        return dict(common, coordinate='affine_signed', scale=bias_scale, boundary=False)
    if name not in ('nodes_time_const', 'edges_syn_strength'):
        raise ValueError(f'Unspecified coordinate group {name}')
    if value > 0:
        return dict(common, coordinate='log_positive', scale=value, boundary=False)
    if value == 0 and name == 'edges_syn_strength' and initialization > 0:
        return dict(common, coordinate='affine_nonnegative', scale=float(initialization), boundary=True)
    raise ValueError(f'Invalid positive-domain baseline {name}[{index}]={value}')


def coordinate_value(entry, z):
    if entry['coordinate'] == 'log_positive':
        return entry['baseline'] * np.exp(z)
    if entry['boundary'] and z < 0:
        raise ValueError('Negative boundary coordinate is infeasible')
    return entry['baseline'] + entry['scale'] * z


def derivative(entry, h, evaluate, f0):
    if entry['boundary']:
        return (-3 * f0 + 4 * evaluate(h) - evaluate(2 * h)) / (2 * h)
    return (evaluate(h) - evaluate(-h)) / (2 * h)


def comparison(J, reference):
    col_norm = np.linalg.norm(reference, axis=0)
    active = col_norm > max(float(col_norm.max()) * 1e-10, 1e-15)
    dot = (J[:, active] * reference[:, active]).sum(axis=0)
    denom = np.linalg.norm(J[:, active], axis=0) * col_norm[active]
    cosine = dot / np.maximum(denom, 1e-300)
    relative = np.linalg.norm(J[:, active] - reference[:, active], axis=0) / col_norm[active]
    return {'relative_frobenius': float(np.linalg.norm(J-reference) / max(np.linalg.norm(reference), 1e-300)),
            'active_columns': int(active.sum()), 'minimum_column_cosine': float(cosine.min()) if len(cosine) else None,
            'median_column_cosine': float(np.median(cosine)) if len(cosine) else None,
            'median_relative_column_difference': float(np.median(relative)) if len(relative) else None,
            'maximum_relative_column_difference': float(relative.max()) if len(relative) else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', choices=['trained', 'initialized'], required=True)
    p.add_argument('--checkpoint', type=Path, help='Official checkpoint file or downloaded ZIP archive')
    p.add_argument('--checkpoint-member', default='results/flow/0000/000/chkpts/chkpt_00000')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=float, nargs='+', default=[.01, .003, .03])
    p.add_argument('--primary-step', type=float, default=.01)
    p.add_argument('--seed', type=int, default=20261004)
    p.add_argument('--synapse-count', type=int, default=0, help='0=all; otherwise seeded subset')
    p.add_argument('--bias-scale', type=float, default=1.)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--dtype', choices=['float64', 'float32'], default='float64')
    args = p.parse_args(argv)
    if args.primary_step not in args.steps or min(args.steps) <= 0 or args.bias_scale <= 0:
        p.error('Positive steps/scales required; primary step must occur in steps')
    if args.state == 'trained' and not args.checkpoint:
        p.error('Trained state requires an explicit checkpoint')
    import torch
    import flyvis
    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.set_default_device(device)
    torch.set_default_dtype(getattr(torch, args.dtype))
    flyvis.device = device
    importlib.import_module('flyvis.network.initialization').device = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    start = time.time()
    net = flyvis.Network().to(dtype=getattr(torch, args.dtype))
    defaults = {n: v.detach().cpu().numpy().copy() for n, v in net.named_parameters() if v.requires_grad}
    checkpoint_info = None
    if args.state == 'trained':
        # torch.save also creates ZIP containers; only the official member
        # distinguishes an ensemble archive from a serialized checkpoint.
        is_ensemble_archive = False
        if zipfile.is_zipfile(args.checkpoint):
            with zipfile.ZipFile(args.checkpoint) as archive:
                is_ensemble_archive = args.checkpoint_member in archive.namelist()
        if is_ensemble_archive:
            with zipfile.ZipFile(args.checkpoint) as archive:
                raw = archive.read(args.checkpoint_member)
            state = torch.load(io.BytesIO(raw), map_location=device, weights_only=False)
            checkpoint_info = {'archive_sha256': sha256(args.checkpoint), 'member': args.checkpoint_member,
                               'member_sha256': hashlib.sha256(raw).hexdigest()}
        else:
            state = torch.load(args.checkpoint, map_location=device, weights_only=False)
            checkpoint_info = {'sha256': sha256(args.checkpoint)}
        net.load_state_dict(state['network'], strict=True)
    free = {n: v for n, v in net.named_parameters() if v.requires_grad}
    original_state = {n: v.detach().clone() for n, v in net.named_parameters()}
    rng = np.random.default_rng(args.seed)
    selected = {n: np.sort(rng.choice(v.numel(), args.synapse_count, replace=False))
                if n == 'edges_syn_strength' and 0 < args.synapse_count < v.numel()
                else np.arange(v.numel()) for n, v in free.items()}
    entries = [chart_entry(n, i, v.detach().view(-1)[i].item(), defaults[n][i], args.bias_scale)
               for n, v in free.items() for i in selected[n]]
    interior = np.array([not e['boundary'] for e in entries])
    xx = np.arange(721)
    mov = np.stack([.5+.5*np.sin(2*np.pi*(xx/40.-f*.1)) for f in range(20)])[None, :, None]
    movie = torch.tensor(mov)
    grids = [64, 32, 16, 8, 6]

    def simulate():
        with torch.no_grad():
            a = net.simulate(movie, .02, initial_state='auto')
            result = a[0, -5:].mean(0).detach().cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite activity')
        return result

    base_raw = simulate()
    bins = {g: np.linspace(0, len(base_raw), g+1).astype(int) for g in grids}

    def read(raw):
        return np.concatenate([np.array([raw[b[k]:b[k+1]].mean() for k in range(g)]) for g,b in bins.items()])

    base = read(base_raw)
    repeats = np.stack([read(simulate()) for _ in range(3)])
    null = repeats-base
    row_scale = np.abs(base)+1e-6
    arrays = {'baseline_raw': base_raw, 'baseline': base, 'output_scales': row_scale,
              'repeated_baseline': repeats, 'interior_mask': interior, 'movie': mov}
    spectra = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for step in args.steps:
        Jraw = np.empty((len(base), len(entries)))
        for col, entry in enumerate(entries):
            v = free[entry['group']].data.view(-1)
            index = entry['index']

            def evaluate(z):
                try:
                    v[index] = coordinate_value(entry, z)
                    return read(simulate())
                finally:
                    v[index] = entry['baseline']

            Jraw[:, col] = derivative(entry, step, evaluate, base)
            if col % 100 == 0:
                print(f'{args.state} h={step:g} column={col}/{len(entries)} elapsed={time.time()-start:.1f}s', flush=True)
        J = Jraw/row_scale[:, None]
        tag = f'{step:g}'
        arrays[f'J_raw_step_{tag}'] = Jraw
        arrays[f'J_scaled_step_{tag}'] = J
        summaries, offset = {}, 0
        for g in grids:
            j = J[offset:offset+g]
            summaries[str(g)] = {'interior': spectrum(j[:, interior]), 'including_boundary': spectrum(j),
                                  'bias_only': spectrum(j[:, [e['group']=='nodes_bias' for e in entries]])}
            offset += g
        spectra[tag] = summaries
        np.savez_compressed(args.output.with_suffix('.npz'), **arrays)
    primary = arrays[f'J_scaled_step_{args.primary_step:g}'][:64]
    validation = {f'{step:g}': {'interior': comparison(arrays[f'J_scaled_step_{step:g}'][:64, interior], primary[:, interior]),
                              'including_boundary': comparison(arrays[f'J_scaled_step_{step:g}'][:64], primary)} for step in args.steps}
    restored = all(torch.equal(v, original_state[n]) for n,v in net.named_parameters())
    if not restored:
        raise AssertionError('A parameter was not exactly restored')
    output = {'experiment': 'flyvis_explicit_state_dimensionless_sensitivity', 'model_state': args.state,
              'trained_checkpoint': checkpoint_info, 'flyvis_version': flyvis.__version__, 'torch_version': torch.__version__,
              'device': args.device, 'dtype': args.dtype, 'seed': args.seed,
              'free_parameter_groups': {n:int(v.numel()) for n,v in free.items()},
              'selected_indices': {n:v.tolist() for n,v in selected.items()}, 'coordinate_manifest': entries,
              'bias_scale': args.bias_scale, 'interior_count': int(interior.sum()), 'boundary_count': int((~interior).sum()),
              'boundary_interpretation': 'right derivatives on nonnegative tangent cone; inclusive Gram is descriptive',
              'primary_step': args.primary_step, 'steps': args.steps,
              'measurement': {'kind':'neural_activity', 'frames':20, 'dt':.02,
                              'initial_state':'1 second grey steady state recomputed for every perturbation',
                              'temporal_average':'last five movie frames', 'readout':'contiguous neuron-index bin means',
                              'bin_boundaries':{str(k):v.tolist() for k,v in bins.items()},
                              'output_scale':'abs(unperturbed bin mean)+1e-6'},
              'null_max_absolute':float(np.max(np.abs(null))),
              'null_relative_frobenius':float(np.linalg.norm(null)/max(np.linalg.norm(base),1e-300)),
              'spectra_by_step':spectra, 'step_validation':validation, 'parameters_restored_exactly':restored,
              'arrays_file':args.output.with_suffix('.npz').name, 'arrays_sha256':sha256(args.output.with_suffix('.npz')),
              'producer_sha256':sha256(__file__), 'elapsed_seconds':time.time()-start,
              'official_sources':['https://turagalab.github.io/flyvis/reference/network/',
                                  'https://turagalab.github.io/flyvis/reference/network_view/']}
    args.output.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    principal = spectra[f'{args.primary_step:g}']['64']['interior']
    print(json.dumps({'state':args.state, 'd90':principal['eff_dim_90'], 'd99':principal['eff_dim_99'],
                      'elapsed_seconds':output['elapsed_seconds']},indent=2),flush=True)


if __name__ == '__main__':
    main()
