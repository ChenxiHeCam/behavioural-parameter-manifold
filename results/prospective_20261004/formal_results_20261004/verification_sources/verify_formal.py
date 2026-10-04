"""Independent numerical audit from retained responses, observations and fits."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import hashlib,json,sys,zipfile
import numpy as np

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[2]/'results/prospective_20261004/formal_results_20261004'
d=json.loads((ROOT/'formal_report.json').read_text());p=d['protocol'];channels=p['channels'];version=p['source_sha256']['persistent_backend.py']
bound=p['parameter_box_abs_log_gain'];sigma=p['noise_sd_mV'];prior=p['prior_sd_log_gain']

def forward(stim,x):
    request={'stimulus':stim,'log_gains':dict(zip(channels,map(float,x)))}
    key=hashlib.sha256(json.dumps({'backend_sha256':version,'request':request},sort_keys=True).encode()).hexdigest()
    meta=json.loads((ROOT/'raw'/f'{key}.json').read_text())
    assert meta['request']==request and meta['backend_sha256']==version
    with np.load(ROOT/'raw'/f'{key}.npz') as z:y=z['response'].astype(float)
    assert y.shape==(80,129) and np.isfinite(y).all()
    return y,meta

def loss(stimuli,observations,x):
    ys=[forward(s,x)[0] for s in stimuli]
    residual=np.concatenate([(y-o).ravel() for y,o in zip(ys,observations)])/sigma
    return float(.5*np.sum(residual**2)+.5*np.sum(np.asarray(x)**2)/prior**2),float(np.sqrt(np.mean(residual**2))*sigma)

def verify_fit(path,stimuli,observations,starts):
    group=json.loads(path.read_text());assert len(group['starts'])==2
    for i,f in enumerate(group['starts']):
        x=np.asarray(f['x']);assert np.max(np.abs(x))<=bound+1e-14
        value,rmse=loss(stimuli,observations,x)
        np.testing.assert_allclose(value,f['objective'],rtol=1e-11,atol=1e-7)
        np.testing.assert_allclose(rmse,f['training_RMSE_mV'],rtol=1e-12,atol=1e-12)
        assert f['noise_matched']==(rmse<=1.05)
        assert f['iterations']==len(f['history'])<=p['max_iterations_per_start']
        previous=loss(stimuli,observations,np.clip(starts[i],-bound,bound))[0]
        for step in f['history']:
            val,err=loss(stimuli,observations,step['x'])
            np.testing.assert_allclose(val,step['objective'],rtol=1e-11,atol=1e-7)
            np.testing.assert_allclose(err,step['training_RMSE_mV'],rtol=1e-12)
            assert val<=previous+1e-7
            previous=val
    assert group['best']==min(group['starts'],key=lambda f:f['objective'])
    return group

with zipfile.ZipFile(ROOT/'source_snapshot.zip') as z:
    for name,sha in p['source_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
assert json.loads((ROOT/'protocol_locked.json').read_text())==p
truths=np.random.default_rng(p['truth_seed']).uniform(-.2,.2,(p['n_independent_truths'],16))
assert len(d['cases'])==len(truths)==12
checks={};primary_rows=[]
for case in d['cases']:
    i=case['case'];folder=ROOT/f'case_{i:02d}';truth=truths[i]
    np.testing.assert_array_equal(truth,case['true_log_gains'])
    assert json.loads((folder/'case_report.json').read_text())==case
    candidates=p['candidate_library'];baseline=p['baseline'];stim=candidates[baseline]
    true_base,_=forward(stim,truth)
    rng=np.random.default_rng(p['noise_seed']+i);base_noise=rng.normal(0,sigma,true_base.shape);extra_noise=rng.normal(0,sigma,true_base.shape)
    with np.load(folder/'baseline_data.npz') as z:
        np.testing.assert_array_equal(z['true_response'],true_base)
        np.testing.assert_array_equal(z['observation'],true_base+base_noise)
        base_obs=z['observation']
    starts=(np.zeros(16),np.random.default_rng(519013+i).normal(0,.1,16))
    baseline_fit=verify_fit(folder/'baseline_fit.json',[stim],[base_obs],starts)
    if case['status']=='baseline_fit_failure':assert not baseline_fit['best']['noise_matched'];checks[f'case_{i:02d}']='retained baseline failure';continue
    if case['status']=='ensemble_failure':checks[f'case_{i:02d}']='retained ensemble failure';continue
    assert case['status']=='complete'
    ensemble_meta=json.loads((folder/'ensemble.json').read_text())
    with np.load(folder/'ensemble.npz') as z:points=z['points'];J=z['base_J']
    assert points.shape==(16,16) and np.max(np.abs(points))<=bound+1e-14
    np.testing.assert_array_equal(points[0],baseline_fit['best']['x'])
    for point,record in zip(points,ensemble_meta['samples']):
        value,_=loss([stim],[base_obs],point)
        np.testing.assert_allclose(value,record['objective'],rtol=1e-11,atol=1e-7)
        assert value<=ensemble_meta['acceptance_objective_threshold']+1e-7
    center=np.asarray(baseline_fit['best']['x']);columns=[]
    for k in range(16):
        a=center.copy();b=center.copy();a[k]=min(bound,a[k]+p['difference_step_log_gain']);b[k]=max(-bound,b[k]-p['difference_step_log_gain'])
        columns.append(((forward(stim,a)[0]-forward(stim,b)[0])/(a[k]-b[k])).ravel())
    np.testing.assert_array_equal(np.column_stack(columns),J)
    np.testing.assert_allclose(np.linalg.eigvalsh(J.T@J/sigma**2+np.eye(16)/prior**2),ensemble_meta['precision_eigenvalues'],rtol=1e-9,atol=1e-7)
    selection=json.loads((folder/'selection_frozen.json').read_text());assert selection==case['selection']
    names=[n for n in candidates if n!=baseline]
    off={'off_mV':-65.,'components':[]};rest=np.stack([forward(off,x)[0].ravel() for x in points])
    scores={};responses={}
    for name in names:
        ys=np.stack([forward(candidates[name],x)[0].ravel() for x in points]);centered=ys-ys.mean(axis=0)
        scores[name]=float(.5*np.linalg.slogdet(np.eye(16)+centered@centered.T/(15*sigma**2))[1]);responses[name]=float(np.sqrt(np.mean((ys-rest)**2)))
        np.testing.assert_allclose(scores[name],selection['ensemble_information_scores'][name],rtol=1e-10,atol=1e-8)
        np.testing.assert_allclose(responses[name],selection['ensemble_response_RMS_mV'][name],rtol=1e-12)
    expected={'complementary_ensemble':max(scores,key=scores.get),'repeat':baseline,'random':str(np.random.default_rng(78543+i).choice(names)),'max_response':max(responses,key=responses.get)}
    assert expected==selection['choices']
    acquisition=json.loads((folder/'acquisition_order.json').read_text());assert acquisition['additional_acquisition_begin_epoch']>=selection['frozen_epoch']
    heldout,meta=forward(p['heldout'],truth);assert meta['saved_epoch']>=selection['frozen_epoch']
    with np.load(folder/'heldout_truth.npz') as z:np.testing.assert_array_equal(z['response'],heldout)
    arm_starts=(np.asarray(baseline_fit['best']['x']),np.asarray(baseline_fit['starts'][1]['x']))
    assert set(t['arm'] for t in case['arms'])==set(p['arms'])
    for arm in case['arms']:
        name=expected[arm['arm']];next_stim=candidates[name];true_next,_=forward(next_stim,truth)
        with np.load(folder/f'{arm["arm"]}_additional_data.npz') as z:
            np.testing.assert_array_equal(z['true_response'],true_next)
            np.testing.assert_array_equal(z['noise'],extra_noise)
            np.testing.assert_array_equal(z['observation'],true_next+extra_noise)
            extra_obs=z['observation']
        group=verify_fit(folder/f'{arm["arm"]}_fit.json',[stim,next_stim],[base_obs,extra_obs],arm_starts)
        np.testing.assert_array_equal(group['best']['x'],arm['estimated_log_gains'])
        prediction,_=forward(p['heldout'],arm['estimated_log_gains'])
        with np.load(folder/f'{arm["arm"]}_heldout_prediction.npz') as z:np.testing.assert_array_equal(z['response'],prediction)
        np.testing.assert_allclose(np.sqrt(np.mean((prediction-heldout)**2)),arm['heldout_RMSE_mV'],rtol=1e-12)
        np.testing.assert_allclose(np.sqrt(np.mean((np.asarray(arm['estimated_log_gains'])-truth)**2)),arm['log_parameter_RMSE'],rtol=1e-12)
        assert arm['noise_matched']==group['best']['noise_matched']
    checks[f'case_{i:02d}']='PASS'
    print(f'case {i:02d}: PASS',flush=True)
result={'status':'PASS','checks':checks,'scope':'Recompute observations, all recorded fitting objectives, parameter bounds and budgets, ensemble gates and Jacobians, selection scores, acquisition order, and heldout/parameter errors from retained outputs. Not fresh forward simulation.','complete_cases':sum(c['status']=='complete' for c in d['cases']),'retained_failure_cases':sum(c['status']!='complete' for c in d['cases'])}
(ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
