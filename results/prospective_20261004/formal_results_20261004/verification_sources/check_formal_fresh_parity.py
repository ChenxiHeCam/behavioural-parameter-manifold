"""Check selected completed formal outputs with fresh model processes."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,subprocess,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'formal_results_20261004'

def check(task):
    case_id,label,stim,x,expected=task
    specification={'tstop':1500,'observable':'motor_voltage','stimulus':stim,'log_gains':dict(zip(CHANNELS,map(float,x)))}
    proc=subprocess.run([sys.executable,str(ROOT/'_prospective_worker.py'),json.dumps(specification)],capture_output=True,text=True,timeout=180,check=True)
    records=[line[7:] for line in proc.stdout.splitlines() if line.startswith('RESULT ')]
    if len(records)!=1:raise RuntimeError('Missing fresh worker response')
    y=np.asarray(json.loads(records[0])['motor'])
    np.testing.assert_array_equal(y,expected)
    np.savez_compressed(OUT/f'fresh_parity_case_{case_id:02d}_{label}.npz',response=y)
    return {'case':case_id,'state':label,'shape':list(y.shape),'max_abs_mV':float(np.max(np.abs(y-expected)))}

if __name__=='__main__':
    report=json.loads((OUT/'formal_report.json').read_text());p=report['protocol'];CHANNELS=p['channels'];tasks=[]
    for case in (report['cases'][0],report['cases'][-1]):
        i=case['case'];folder=OUT/f'case_{i:02d}'
        with np.load(folder/'baseline_data.npz') as z:expected=z['true_response']
        tasks.append((i,'baseline_truth',p['candidate_library'][p['baseline']],case['true_log_gains'],expected))
        if case['status']=='complete':
            for label in ('complementary_ensemble','repeat'):
                arm=next(a for a in case['arms'] if a['arm']==label)
                with np.load(folder/f'{label}_heldout_prediction.npz') as z:expected=z['response']
                tasks.append((i,label,p['heldout'],arm['estimated_log_gains'],expected))
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(check,tasks))
    state={'status':'PASS','fresh_forward_checks':results,'worker_sha256':hashlib.sha256((ROOT/'_prospective_worker.py').read_bytes()).hexdigest(),'scope':'Fresh model-process checks at the first/last independent truths and fitted complementary/repeat states; not all formal rollouts.'}
    (OUT/'fresh_forward_parity.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state,indent=2))
