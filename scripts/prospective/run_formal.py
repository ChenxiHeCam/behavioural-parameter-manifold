"""Prospective model experiment with fitted ensembles and nonlinear inference.

The algorithm is fixed before test truths are drawn. Failed cases remain in the
report. All four arms have the same observed-data and optimization budgets.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,time
from pathlib import Path
import numpy as np
from nonlinear_engine import Backend,fit,fitted_ensemble,select_from_ensemble,CHANNELS,BOUND,PRIOR_SD,H,NOISE
from run_pilot import GROUPS,stimulus

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'formal_results_20261004'
ARMS=('complementary_ensemble','repeat','random','max_response')

def write(path,value):
    temp=Path(str(path)+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def locked_protocol():
    candidates={f'{g}_step_{on:+g}':stimulus(g,on) for g in GROUPS for on in (-5.,5.)}
    candidates.update({f'{g}_ramp_+5':stimulus(g,5.,'ramp') for g in ('olfactory','chemical')})
    heldout={'off_mV':-65.,'components':[
        {'cells':GROUPS['ury'],'shape':'ramp','start_ms':200,'end_ms':950,'on_mV':2.},
        {'cells':GROUPS['posterior'],'shape':'step','start_ms':1050,'end_ms':1250,'on_mV':-2.}]}
    files=('run_formal.py','nonlinear_engine.py','persistent_backend.py','_prospective_worker.py')
    return {'study':'prospective BAAIWorm synthetic-data comparison','n_independent_truths':12,'noise_realizations_per_truth':1,'channels':CHANNELS,'truth_seed':202610041,'truth_distribution':'independent uniform log gains [-0.2,0.2] about published reference','noise_seed':680019,'noise_sd_mV':NOISE,'prior_sd_log_gain':PRIOR_SD,'parameter_box_abs_log_gain':BOUND,'baseline':'olfactory_step_-5','candidate_library':candidates,'heldout':heldout,'heldout_used_for_selection':False,'duration_ms':1500,'dt_ms':5/3,'output_shape':[80,129],'observed_budget_per_arm':'one baseline plus one additional full voltage trajectory','shared_randomness':'baseline observation shared across arms; same independent additional noise array across arms; independent arrays across truths','fit':'bounded trust-region nonlinear least squares with Gaussian log-gain regularization','difference_step_log_gain':H,'fit_initial_trust_radius_L2':.05,'max_iterations_per_start':50,'starts_per_baseline_and_arm':2,'noise_matching_RMSE_threshold_mV':1.05,'ensemble_size':16,'ensemble':'baseline-compatible Laplace proposals with fixed scales 1, 0.5, 0.25 and nonlinear objective gate; not exact posterior samples','ensemble_objective_increment_gate':13.148,'arms':list(ARMS),'selection':'Gaussian output-ensemble information score computed on candidate predictions, without truth or heldout outputs','primary_endpoint':'nonlinear heldout prediction RMSE mV','secondary_endpoint':'log-parameter RMSE and training noise-match failure rates','failure_policy':'retain all baseline/ensemble failures; report conditional comparison and denominators; retain arm fitting failures in endpoint summaries','statistical_unit':'independent generating parameter vector, n=12; not voltage samples','uncertainty':'paired bootstrap of independent truths, 10000 draws, fixed seed 921711','scope':'synthetic Gaussian observations of this point-neuron model and these interventions; no animal validation or general identifiability theorem','source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}

def fit_saved(path,backend,stimuli,observations,starts):
    if path.exists():return json.loads(path.read_text())
    fits=[]
    for start in starts:fits.append(fit(backend,stimuli,observations,start,max_iterations=50))
    result={'starts':fits,'best':min(fits,key=lambda f:f['objective'])}
    write(path,result)
    return result

def run_case(index,truth,backend,protocol):
    folder=OUT/f'case_{index:02d}';folder.mkdir(exist_ok=True)
    if (folder/'case_report.json').exists():return json.loads((folder/'case_report.json').read_text())
    began=time.time();baseline=protocol['baseline'];candidates=protocol['candidate_library'];stim=candidates[baseline]
    rng=np.random.default_rng(protocol['noise_seed']+index)
    true_base=backend.one(stim,truth);base_obs=true_base+rng.normal(0,NOISE,true_base.shape)
    extra_noise=rng.normal(0,NOISE,true_base.shape)
    np.savez_compressed(folder/'baseline_data.npz',observation=base_obs,noise=base_obs-true_base,true_response=true_base)
    starts=(np.zeros(16),np.random.default_rng(519013+index).normal(0,.1,16))
    baseline_fit=fit_saved(folder/'baseline_fit.json',backend,[stim],[base_obs],starts)
    best=baseline_fit['best'];base_summary={k:best[k] for k in ('objective','training_RMSE_mV','noise_matched','iterations','termination')}
    print(f'case {index:02d} baseline '+json.dumps(base_summary),flush=True)
    if not best['noise_matched']:
        result={'case':index,'status':'baseline_fit_failure','baseline_fit':base_summary,'true_log_gains':truth.tolist(),'elapsed_sec':time.time()-began}
        write(folder/'case_report.json',result);return result
    ensemble_path=folder/'ensemble.npz'
    if ensemble_path.exists():
        with np.load(ensemble_path) as z:points=z['points']
        ensemble_meta=json.loads((folder/'ensemble.json').read_text())
    else:
        try:points,ensemble_meta=fitted_ensemble(backend,stim,base_obs,best,37019+index,size=16)
        except RuntimeError as e:
            result={'case':index,'status':'ensemble_failure','reason':str(e),'baseline_fit':base_summary,'true_log_gains':truth.tolist(),'elapsed_sec':time.time()-began}
            write(folder/'case_report.json',result);return result
        J=ensemble_meta.pop('base_J')
        np.savez_compressed(ensemble_path,points=points,base_J=J)
        write(folder/'ensemble.json',ensemble_meta)
    frozen=folder/'selection_frozen.json'
    if frozen.exists():selection=json.loads(frozen.read_text())
    else:
        choices,scores=select_from_ensemble(backend,candidates,baseline,points,78543+index)
        selection={'choices':choices,**scores,'frozen_epoch':time.time(),'input':'baseline-fitted parameter ensemble only','heldout_and_truth_candidate_outputs_unavailable_to_selection':True}
        write(frozen,selection)
    print(f'case {index:02d} choices '+json.dumps(selection['choices']),flush=True)
    # Only after selection: acquire additional observations and heldout truth.
    true_heldout=backend.one(protocol['heldout'],truth)
    np.savez_compressed(folder/'heldout_truth.npz',response=true_heldout)
    write(folder/'acquisition_order.json',{'selection_frozen_epoch':selection['frozen_epoch'],'additional_acquisition_begin_epoch':time.time(),'status':'PASS','scope':'Fresh test case; selections do not receive generating gains or heldout predictions.'})
    results=[]
    # Identical starting vectors and iteration budgets in all arms.
    arm_starts=(np.asarray(best['x']),np.asarray(baseline_fit['starts'][1]['x']))
    for arm in ARMS:
        selected=selection['choices'][arm];next_stim=candidates[selected]
        extra_true=backend.one(next_stim,truth);extra_obs=extra_true+extra_noise
        np.savez_compressed(folder/f'{arm}_additional_data.npz',observation=extra_obs,noise=extra_noise,true_response=extra_true)
        fitted=fit_saved(folder/f'{arm}_fit.json',backend,[stim,next_stim],[base_obs,extra_obs],arm_starts)
        final=fitted['best'];x=np.asarray(final['x']);prediction=backend.one(protocol['heldout'],x)
        np.savez_compressed(folder/f'{arm}_heldout_prediction.npz',response=prediction)
        result={'arm':arm,'selected':selected,'estimated_log_gains':x.tolist(),'log_parameter_RMSE':float(np.sqrt(np.mean((x-truth)**2))),'heldout_RMSE_mV':float(np.sqrt(np.mean((prediction-true_heldout)**2))),'training_RMSE_mV':final['training_RMSE_mV'],'noise_matched':final['noise_matched'],'iterations':final['iterations'],'termination':final['termination']}
        results.append(result)
        print(f'case {index:02d} arm '+json.dumps(result),flush=True)
    result={'case':index,'status':'complete','baseline_fit':base_summary,'true_log_gains':truth.tolist(),'selection':selection,'arms':results,'ensemble_metadata':ensemble_meta,'elapsed_sec':time.time()-began}
    write(folder/'case_report.json',result);return result

def main():
    OUT.mkdir(exist_ok=True)
    protocol=locked_protocol();protocol_path=OUT/'protocol_locked.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())!=protocol:raise RuntimeError('Locked protocol or code has changed')
    else:write(protocol_path,protocol)
    parity=json.loads((ROOT/'formal_development/persistent_backend_parity.json').read_text())
    if parity['status']!='PASS' or parity['backend_sha256']!=protocol['source_sha256']['persistent_backend.py']:raise RuntimeError('Backend parity missing or stale')
    development=json.loads((ROOT/'formal_development/nonlinear_development_report.json').read_text())
    if development['status']!='PASS' or development['engine_sha256']!=protocol['source_sha256']['nonlinear_engine.py']:raise RuntimeError('Nonlinear development missing or stale')
    rng=np.random.default_rng(protocol['truth_seed'])
    truths=rng.uniform(-.2,.2,(protocol['n_independent_truths'],16))
    backend=Backend(OUT,n_workers=24);started=time.time()
    records=[]
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending={pool.submit(run_case,i,x,backend,protocol):i for i,x in enumerate(truths)}
            for future in as_completed(pending):
                i=pending[future]
                try:record=future.result()
                except Exception as e:
                    write(OUT/f'case_{i:02d}/execution_failure.json',{'error':type(e).__name__+': '+str(e)})
                    raise
                records.append(record)
                write(OUT/'RUN_STATUS.json',{'status':'running','completed_cases':sorted(r['case'] for r in records),'total_cases':len(truths),'retained_rollouts':backend.count,'elapsed_sec':time.time()-started})
                print(f'CASE FINISHED {i:02d} {record["status"]} {len(records)}/{len(truths)}',flush=True)
    finally:backend.close()
    records.sort(key=lambda r:r['case'])
    result={'status':'complete','protocol':protocol,'cases':records,'elapsed_sec':time.time()-started,'retained_rollouts':len(list((OUT/'raw').glob('*.npz'))),'interpretation':'Prospective synthetic-data experiment for the frozen algorithm and model; report fit failures and independent-truth uncertainty before making any performance claim.'}
    write(OUT/'formal_report.json',result)
    write(OUT/'RUN_STATUS.json',{'status':'complete','completed_cases':list(range(len(truths))),'total_cases':len(truths),'retained_rollouts':result['retained_rollouts'],'running_experiments':[]})
    print('FORMAL COMPLETE '+json.dumps({'cases':len(records),'rollouts':result['retained_rollouts'],'elapsed_sec':result['elapsed_sec']}),flush=True)

if __name__=='__main__':main()
