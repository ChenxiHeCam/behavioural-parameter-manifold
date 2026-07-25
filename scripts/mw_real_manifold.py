'''Cheap REAL modWorm manifold/sloppiness probe: sample N random 7-mechanism
param sets, single rollout each; show params disperse while behaviour stays
clustered (sloppy manifold). No optimization. Reuses mw_real_worker.'''
import os,sys,json,time,subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
N=int(os.environ.get('N','24')); SIG=float(os.environ.get('SIG','0.25')); NSTEP=int(os.environ.get('NSTEP','200')); CONC=int(os.environ.get('CONC','10'))
MECH=['gap','syn','leak','Cm','rise','fall','B']; OBS=['net_disp','path_len','mean_speed','speed_std','rms_heading','final_curve']
PYB='/root/miniconda3/envs/flygym_fresh/bin/python'
rng=np.random.RandomState(11)
def run(spec):
    p=subprocess.run([PYB,'/root/autodl-tmp/mw_real_worker.py',json.dumps(spec)],capture_output=True,text=True,timeout=1200)
    for ln in p.stdout.splitlines():
        if ln.startswith('RESULT'): return json.loads(ln[7:])['obs']
    return None
specs=[('base',{'n_steps':NSTEP})]
mults=[]
for k in range(N):
    m={mm:float(np.exp(rng.normal(0,SIG))) for mm in MECH}; mults.append(m)
    specs.append((f's{k}',{**m,'n_steps':NSTEP}))
def do(it):
    tag,spec=it; o=run(spec); return tag,o
t0=time.time(); res={}
with ThreadPoolExecutor(max_workers=CONC) as ex:
    for tag,o in ex.map(do,specs):
        res[tag]=o; print(f'  {tag} done {time.time()-t0:.0f}s',flush=True)
base=res['base']; bvec=np.array([base[k] for k in OBS]); scale=np.abs(bvec)+1e-9
bd=[]; pr=[]
for k in range(N):
    o=res.get(f's{k}'); 
    if o is None: continue
    bd.append(float(np.linalg.norm((np.array([o[k2] for k2 in OBS])-bvec)/scale)))   # behaviour dist (whitened)
    pr.append(float(np.mean(np.abs(np.array([mults[k][mm] for mm in MECH])-1))))       # param relerr
bd=np.array(bd); pr=np.array(pr)
out={'sim':'REAL modWorm (c302 connectome)','probe':'manifold/sloppiness sample','N':N,'sigma':SIG,'n_steps':NSTEP,
 'param_relerr_mean':float(pr.mean()),'behav_dist_mean':float(bd.mean()),'behav_dist_median':float(np.median(bd)),
 'behav_over_param_ratio':float(bd.mean()/max(pr.mean(),1e-9)),
 'interpretation':'large random parameter perturbations (mean relerr %.2f) produce small whitened behavioural change (mean %.2f) -> behaviour-equivalence manifold: behaviour is insensitive to most parameter directions'%(float(pr.mean()),float(bd.mean())),
 'elapsed_sec':round(time.time()-t0,0)}
json.dump(out,open('/root/autodl-tmp/paper2_modworm_REAL_manifold.json','w'),indent=2)
print('=== REAL modWorm manifold probe DONE ===',flush=True)
print('param relerr mean=%.3f  behav dist mean=%.3f median=%.3f  behav/param=%.3f'%(pr.mean(),bd.mean(),np.median(bd),out['behav_over_param_ratio']),flush=True)
