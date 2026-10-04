"""Execute real NEURON rollouts to verify SLO mechanism and readout dispatch."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def main():
    worker = Path(__file__).with_name('_baai_worker.py')
    out = Path(os.environ['PAPER2_OUTPUT_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    specs = {'baseline': {'conn': -1, 'tstop': 100}}
    for mechanism in ('slo1_egl19', 'slo1_unc2', 'slo2_egl19', 'slo2_unc2'):
        specs[mechanism] = {'conn': -1, 'tstop': 100, 'channel': 'gb'+mechanism.split('_')[0],
                            'mech': mechanism, 'sign': 1, 'delta': .5}
    for observable in ('muscle_preactivation', 'muscle_activation'):
        specs[observable] = {'conn': -1, 'tstop': 100, 'observable': observable}
    specs['missing_mechanism'] = {'conn': -1, 'tstop': 100, 'channel': 'gbnotreal',
                                  'mech': 'notreal', 'sign': 1, 'delta': .5}
    def run(item):
        key, spec = item
        p = subprocess.run([sys.executable, str(worker), json.dumps(spec)],
                           capture_output=True, text=True, timeout=120)
        (out/(key+'.log')).write_text(p.stdout+p.stderr)
        for line in p.stdout.splitlines():
            if line.startswith('RESULT '):
                result = json.loads(line[7:])
                return key, result
        if key == 'missing_mechanism' and p.returncode != 0:
            return key, {'expected_error': True}
        raise RuntimeError(f'Smoke failed: {key}, exit {p.returncode}')
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = dict(pool.map(run, specs.items()))
    baseline = np.asarray(results['baseline']['motor'])
    summary = {'worker_sha256': hashlib.sha256(worker.read_bytes()).hexdigest(),
               'specs': specs, 'checks': {}}
    for key, result in results.items():
        if key in ('baseline', 'missing_mechanism'):
            continue
        value = np.asarray(result['motor'])
        if key.startswith('slo'):
            assert result['n_conductance_segments_touched'] > 0
            assert not np.array_equal(value, baseline)
            summary['checks'][key] = {'segments_touched': result['n_conductance_segments_touched'],
                                      'motor_output_difference_norm': float(np.linalg.norm(value-baseline))}
        else:
            assert value.shape[0] == 96
            if key == 'muscle_activation':
                assert value.min() >= 0 and value.max() <= .8
            summary['checks'][key] = {'shape': list(value.shape), 'min': float(value.min()),
                                      'max': float(value.max())}
    assert results['missing_mechanism']['expected_error']
    summary['checks']['missing_mechanism'] = 'failed explicitly as expected'
    (out/'worker_smoke.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary['checks'], indent=2))


if __name__ == '__main__':
    main()
