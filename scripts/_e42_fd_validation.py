"""Matched baseline repeats and sampled finite-step checks for the new E42."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np

out=Path(os.environ['PAPER2_OUTPUT_DIR'])
worker=Path(__file__).with_name('_baai_worker.py')
with np.load(out/'E42_motor_voltage_matrices.npz') as z:
    J=z['J']; base=z['baseline']; scale=z['output_scale']
norm=np.linalg.norm(J,axis=0)
order=np.argsort(norm)
samples={'strongest':int(order[-1]),'median':int(order[len(order)//2]),
         'lower_quartile':int(order[len(order)//4]),
         'weakest_nonzero':int(order[np.flatnonzero(norm[order]>norm.max()*1e-12)[0]])}
steps=(.05,.1,.25,.5)
jobs=[('repeat0',{'conn':-1,'sign':0,'delta':.5,'tstop':2000}),
      ('repeat1',{'conn':-1,'sign':0,'delta':.5,'tstop':2000})]
for label,connection in samples.items():
    for step in steps:
        for sign in (1,-1):
            jobs.append((f'{label}_{step}_{sign:+d}',{'conn':connection,'sign':sign,'delta':step,'tstop':2000}))
def run(item):
    name,spec=item
    p=subprocess.run([sys.executable,str(worker),json.dumps(spec)],capture_output=True,text=True,timeout=120)
    for line in p.stdout.splitlines():
        if line.startswith('RESULT '):
            return name,np.asarray(json.loads(line[7:])['motor'],dtype=np.float64)
    raise RuntimeError(f'FD validation failed {name}, exit{p.returncode}: {p.stderr[-300:]}')
with ThreadPoolExecutor(max_workers=12) as pool:
    responses=dict(pool.map(run,jobs))
report={'experiment':'matched_E42_baseline_repeat_and_sampled_steps',
        'n_sampled_connections':len(samples),'n_total_connections':J.shape[1],
        'samples':samples,'steps':steps,'n_rollouts':len(jobs),
        'scope':'four sampled columns only; not all3076columns and not a full-spectrum step sweep',
        'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),
        'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'baseline_exact_match_to_main':{k:bool(np.array_equal(v,base)) for k,v in responses.items() if k.startswith('repeat')},
        'baseline_repeat_max_abs_difference':float(np.max(np.abs(responses['repeat0']-responses['repeat1']))),
        'columns':{}}
arrays={**responses,'output_scale':scale}
for label,connection in samples.items():
    derivatives={step:((responses[f'{label}_{step}_+1']-responses[f'{label}_{step}_-1'])/(2*step)/scale).ravel() for step in steps}
    small=derivatives[.05]
    smallnorm=np.linalg.norm(small)
    rows=[]
    for step,d in derivatives.items():
        size=np.linalg.norm(d)
        rows.append({'delta':step,'normalized_derivative_norm':float(size),
                     'relative_change_from_delta0_05':float(np.linalg.norm(d-small)/smallnorm) if smallnorm else None,
                     'direction_cosine_to_delta0_05':float(np.dot(d,small)/(size*smallnorm)) if size and smallnorm else None,
                     'normalized_symmetric_response_norm':float(size*2*step)})
        arrays[f'derivative_{label}_{step}']=d
    report['columns'][label]={'connection_index':connection,'full_probe_delta0_5_norm':float(norm[connection]),
                              'delta0_5_exact_match_to_main_J':bool(np.array_equal(derivatives[.5],J[:,connection])),
                              'steps':rows}
np.savez_compressed(out/'E42_fd_validation_arrays.npz',**arrays)
(out/'E42_fd_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
