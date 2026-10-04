"""Bounded feasibility pilot, not the confirmatory recovery experiment.

Computes equal-duration stimulus Jacobians at the published operating point.
Synthetic truths are generated only after candidate selection is frozen. The
design uses this fixed reference, not an ensemble fitted to baseline data, so
results cannot support the manuscript's proposed prospective inference claim.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import concurrent.futures as cf
import hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'pilot_results'
CHANNELS=['cca1','egl19','egl2','egl36','irk','kcnl','kqt3','kvs1','nca','shk1','shl1','slo1_egl19','slo1_unc2','slo2_egl19','slo2_unc2','unc2']
GROUPS={'olfactory':['AWAL','AWAR','AWCL','AWCR'],'chemical':['ASKL','ASKR','PHAL','PHAR'],'anterior':['ALNL','ALNR'],'posterior':['PLML'],'ury':['URYDL','URYDR','URYVL','URYVR']}
DT=5/3
NOISE=1.0
PRIOR_SD=.35

def stimulus(group,on,shape='step',start=250,end=1000):
    return {'off_mV':-65.,'components':[{'cells':GROUPS[group],'shape':shape,'start_ms':start,'end_ms':end,'on_mV':on}]}

def call(spec,original=False):
    key=hashlib.sha256(json.dumps({'spec':spec,'original':original},sort_keys=True).encode()).hexdigest()
    filename=OUT/'raw'/f'{key}.npz'
    if filename.exists():
        with np.load(filename) as z:return z['response']
    worker=ROOT/('_baai_worker.py' if original else '_prospective_worker.py')
    t=time.time()
    p=subprocess.run([sys.executable,str(worker),json.dumps(spec)],capture_output=True,text=True,timeout=180)
    lines=[line[7:] for line in p.stdout.splitlines() if line.startswith('RESULT ')]
    if p.returncode or len(lines)!=1:
        (OUT/'raw'/f'{key}.failure.txt').write_text(p.stdout+'\n'+p.stderr)
        raise RuntimeError(f'Rollout failed {key}: {p.stderr[-600:]}')
    result=json.loads(lines[0]); y=np.asarray(result['motor'],dtype=float)
    if y.shape!=(80,129) or not np.isfinite(y).all():raise ValueError('Invalid motor output')
    np.savez_compressed(filename,response=y)
    (OUT/'raw'/f'{key}.json').write_text(json.dumps({'spec':spec,'original':original,'metadata':{k:v for k,v in result.items() if k!='motor'},'wall_sec':time.time()-t},indent=2))
    return y

def spec(stim,gains=None):return {'tstop':1500,'observable':'motor_voltage','stimulus':stim,'log_gains':gains or {}}

def metrics(H):
    vals=np.linalg.eigvalsh(H)[::-1]
    positive=np.maximum(vals,0)
    mass=np.cumsum(positive)/positive.sum()
    return {'eigenvalues':vals.tolist(),'d90':int(np.searchsorted(mass,.9)+1),'d99':int(np.searchsorted(mass,.99)+1),'participation_ratio':float(positive.sum()**2/(positive@positive))}

def jacobian(stim,h=.1):
    tasks=[spec(stim,{channel:sign*h}) for channel in CHANNELS for sign in (1,-1)]
    with cf.ThreadPoolExecutor(max_workers=8) as pool:ys=list(pool.map(call,tasks))
    return np.column_stack([((ys[2*i]-ys[2*i+1])/(2*h)).ravel() for i in range(len(CHANNELS))])

def main():
    OUT.mkdir(exist_ok=True);(OUT/'raw').mkdir(exist_ok=True)
    started=time.time()
    historical=call({'tstop':1500,'observable':'motor_voltage'},True)
    with np.load(ROOT/'reference_E45.npz') as z:ref=z['baseline'].astype(float)
    parity={'shape':list(historical.shape),'max_abs_mV':float(np.max(np.abs(historical.astype(np.float32)-ref)))}
    (OUT/'runtime_parity.json').write_text(json.dumps(parity,indent=2))
    print('Runtime parity '+json.dumps(parity),flush=True)
    if parity['max_abs_mV']>1e-4:raise RuntimeError('Historical baseline runtime mismatch')
    candidates={f'{g}_step_{on:+g}':stimulus(g,on) for g in GROUPS for on in (-5.,5.)}
    candidates.update({f'{g}_ramp_+5':stimulus(g,5.,'ramp') for g in ('olfactory','chemical')})
    baseline='olfactory_step_-5'
    heldout={'off_mV':-65.,'components':[
        {'cells':GROUPS['chemical'],'shape':'ramp','start_ms':150,'end_ms':850,'on_mV':0.},
        {'cells':GROUPS['anterior'],'shape':'step','start_ms':900,'end_ms':1100,'on_mV':5.}]}
    protocol={'phase':'feasibility_pilot','channels':CHANNELS,'candidates':candidates,'baseline':baseline,'heldout':heldout,'tstop_ms':1500,'dt_ms':DT,'noise_sd_mV':NOISE,'prior_sd_log_gain':PRIOR_SD,'h_log_gain':.1,'n_workers':8,'limitation':'Design at published reference, not a baseline-fitted parameter ensemble; synthetic noise and model stimuli.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    matrices={}; responses={}; rest=call(spec({'off_mV':-65.,'components':[]}))
    summaries={}
    for name,stim in candidates.items():
        y=call(spec(stim));J=jacobian(stim);H=J.T@J/NOISE**2
        responses[name]=y;matrices[name]=J
        summaries[name]={**metrics(H),'response_distance_from_unstimulated_mV':float(np.linalg.norm(y-rest))}
        np.savez_compressed(OUT/f'{name}.npz',J_raw_mV=J,G=H,baseline=y,eigenvalues=np.linalg.eigvalsh(H)[::-1])
        print(name+' '+json.dumps({k:summaries[name][k] for k in ('d90','d99','participation_ratio')}),flush=True)
    small=jacobian(candidates[baseline],.03);large=matrices[baseline]
    repeat=call({**spec(candidates[baseline]),'repeat_id':1})
    fd={'h_reference':.1,'h_small':.03,'relative_J_Frobenius_difference':float(np.linalg.norm(small-large)/np.linalg.norm(small)),'repeat_max_abs_mV':float(np.max(np.abs(repeat-responses[baseline]))),'column_relative_difference':(np.linalg.norm(small-large,axis=0)/np.maximum(np.linalg.norm(small,axis=0),1e-15)).tolist()}
    prior=np.eye(16)/PRIOR_SD**2
    H0=large.T@large/NOISE**2
    score={name:float(np.linalg.slogdet(prior+H0+J.T@J/NOISE**2)[1]) for name,J in matrices.items() if name!=baseline}
    selected=max(score,key=score.get)
    eligible=list(score)
    random_choice=str(np.random.default_rng(2907).choice(eligible))
    max_response=max(eligible,key=lambda n:summaries[n]['response_distance_from_unstimulated_mV'])
    selection={'complementary_logdet':selected,'repeat':baseline,'random_seed_2907':random_choice,'max_response':max_response,'scores_logdet':score,'frozen_before_hidden_truths':True}
    (OUT/'selection_frozen.json').write_text(json.dumps(selection,indent=2))
    print('Selection frozen '+json.dumps({k:v for k,v in selection.items() if isinstance(v,str)}),flush=True)
    # No heldout forward call is made until the candidate choice is committed.
    yh=call(spec(heldout));Jh=jacobian(heldout)
    rng=np.random.default_rng(89041)
    truths=rng.uniform(-.2,.2,(3,16))
    trials=[]
    for truth_id,truth in enumerate(truths):
        gains=dict(zip(CHANNELS,map(float,truth)))
        true_h=call(spec(heldout,gains))
        needed=set(selection[k] for k in ('complementary_logdet','repeat','random_seed_2907','max_response'))|{baseline}
        noisy={name:call(spec(candidates[name],gains)).ravel()+rng.normal(0,NOISE,large.shape[0]) for name in sorted(needed)}
        repeat_noise=call(spec(candidates[baseline],gains)).ravel()+rng.normal(0,NOISE,large.shape[0])
        for arm in ('complementary_logdet','repeat','random_seed_2907','max_response'):
            name=selection[arm];J1=matrices[name]
            obs1=repeat_noise if arm=='repeat' else noisy[name]
            estimate=np.linalg.solve(prior+H0+J1.T@J1/NOISE**2,(large.T@(noisy[baseline]-responses[baseline].ravel())+J1.T@(obs1-responses[name].ravel()))/NOISE**2)
            prediction=call(spec(heldout,dict(zip(CHANNELS,map(float,estimate)))))
            trials.append({'truth_id':truth_id,'arm':arm,'selected':name,'true_log_gains':truth.tolist(),'estimated_log_gains':estimate.tolist(),'log_parameter_RMSE':float(np.sqrt(np.mean((estimate-truth)**2))),'heldout_RMSE_mV':float(np.sqrt(np.mean((prediction-true_h)**2))),'heldout_linearization_error_RMSE_mV':float(np.sqrt(np.mean((yh.ravel()+Jh@truth-true_h.ravel())**2)))})
    report={'status':'complete','interpretation':'Exploratory pilot only; 3 truths and 1 noise realization each, no confirmatory significance or fitted-ensemble design claim.','runtime_parity':parity,'protocol':protocol,'candidates':summaries,'finite_difference_check':fd,'selection':selection,'trials':trials,'elapsed_sec':time.time()-started,'worker_sha256':hashlib.sha256((ROOT/'_prospective_worker.py').read_bytes()).hexdigest()}
    (OUT/'pilot_report.json').write_text(json.dumps(report,indent=2))
    print('PILOT COMPLETE elapsed_sec='+str(round(time.time()-started,1)),flush=True)

if __name__=='__main__':main()
