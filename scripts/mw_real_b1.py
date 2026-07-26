'''REAL modWorm per-mechanism B1: behavioural elasticity Hessian over 7 named
biophysical mechanisms. H=J^T J from elasticity Jacobian (central diff in log-mult).'''
import os,sys,json,time,subprocess,math
import numpy as np
from concurrent.futures import ThreadPoolExecutor
DELTA=float(os.environ.get('DELTA','0.25')); NSTEP=int(os.environ.get('NSTEP','250')); CONC=int(os.environ.get('CONC','14'))
MECH=['gap','syn','leak','Cm','rise','fall','B']
NAME={'gap':'gap-junction conductance','syn':'synaptic conductance','leak':'leak conductance',
 'Cm':'membrane capacitance','rise':'synaptic rise tau','fall':'synaptic fall tau','B':'sigmoid gain'}
OBS=['net_disp','path_len','mean_speed','speed_std','rms_heading','final_curve']
PY_BIN='/root/miniconda3/envs/flygym_fresh/bin/python'
def run_spec(spec):
    p=subprocess.run([PY_BIN,'/root/autodl-tmp/mw_real_worker.py',json.dumps(spec)],capture_output=True,text=True,timeout=1200)
    for ln in p.stdout.splitlines():
        if ln.startswith('RESULT'): return json.loads(ln[7:])['obs']
    return None
jobs=[('base',{'n_steps':NSTEP})]
for m in MECH:
    jobs.append((m+'+',{m:1+DELTA,'n_steps':NSTEP})); jobs.append((m+'-',{m:1-DELTA,'n_steps':NSTEP}))
def do(j):
    tag,spec=j; t0=time.time(); obs=run_spec(spec)
    print(f'  {tag} {time.time()-t0:.0f}s {None if obs is None else round(obs["net_disp"],4)}',flush=True)
    return tag,obs
t0=time.time(); res={}
with ThreadPoolExecutor(max_workers=CONC) as ex:
    for tag,obs in ex.map(do,jobs): res[tag]=obs
base=res['base']; bvec=np.array([base[k] for k in OBS]); scale=np.abs(bvec)+1e-9
dln=math.log(1+DELTA)-math.log(1-DELTA)
J=np.zeros((len(OBS),len(MECH))); elast={}
for j,m in enumerate(MECH):
    op,om=res.get(m+'+'),res.get(m+'-')
    if op is None or om is None: continue
    col=((np.array([op[k] for k in OBS])-np.array([om[k] for k in OBS]))/scale)/dln
    J[:,j]=col; elast[m]=float(np.linalg.norm(col))
H=J.T@J; ev=np.maximum(np.linalg.eigvalsh(H)[::-1],0)
cs=np.cumsum(ev)/max(ev.sum(),1e-30)
eff90=int(np.searchsorted(cs,0.9)+1); eff99=int(np.searchsorted(cs,0.99)+1)
pr=float((ev.sum()**2)/np.maximum((ev**2).sum(),1e-30))
order=sorted(elast,key=lambda k:-elast[k])
out={'sim':'REAL modWorm (c302 connectome, 279 neurons, coupled neural+body)','probe':'per-mechanism elasticity Hessian',
 'connectome':'OpenWorm/c302 White-Varshney-Cook curation mapped to modWorm 279','delta':DELTA,'n_steps':NSTEP,
 'mechanisms':MECH,'names':NAME,'base_obs':base,'eigenvalues':ev.tolist(),
 'eff_dim_90':eff90,'eff_dim_99':eff99,'participation_ratio':pr,
 'per_mechanism_elasticity':elast,
 'rank_stiff_to_sloppy':[(m,NAME[m],round(elast[m],4)) for m in order],'elapsed_sec':round(time.time()-t0,0)}
json.dump(out,open('/root/autodl-tmp/modworm_REAL_b1.json','w'),indent=2)
print('=== REAL modWorm B1 DONE ===',flush=True)
print(f'eff_dim 90%={eff90} 99%={eff99}/7 PR={pr:.2f}',flush=True)
print('STIFF->SLOPPY:',[(m,round(elast[m],3)) for m in order],flush=True)
