'''B6b: worm UNION proof at the CELL-CLASS level (where the complementarity lives).
Perturb the excitability (gpas) of each neuron class under chemotaxis (food close) vs plain
locomotion (food far), build a per-cell-class GN Hessian for each behaviour, and test whether the
UNION (H_chemo + H_loco) has higher effective dimension than either single behaviour -- i.e.
chemotaxis makes the chemosensory classes stiff, locomotion the motor classes, complementarily.
Run: xvfb-run -a python3 _baai_b6b_cellunion.py'''
import os, json, time, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor

OBS = ['net_disp', 'path_len', 'mean_speed', 'speed_std', 'rms_heading', 'fwd_world']
# neuron classes by name prefix (model skips any absent in its cell set)
CLASSES = {
  'AWA': ['AWA'], 'AWC': ['AWC'], 'ASE': ['ASE'], 'ASH': ['ASH'],     # chemosensory
  'AIY': ['AIY'], 'AIZ': ['AIZ'], 'RIA': ['RIA'], 'AIB': ['AIB'],     # interneurons
  'VA': ['VA'], 'VB': ['VB'], 'DA': ['DA'], 'DB': ['DB'],             # ventral/dorsal motor
  'VD': ['VD'], 'DD': ['DD'],                                          # GABA motor
}
CLABELS = list(CLASSES.keys())
DELTA = 0.25; NSTEP = 120
WORKER = '/root/_baai_perturb_worker_orig.py'
FOOD_CLOSE = [1.8275, -0.0276, -0.3082]; FOOD_FAR = [1.5, 0.2, -0.5]

def spec(cls, factor, food):
    s = {'syn':1.0,'gj':1.0,'wout':1.0,'ion':{},'passive':{},'n_steps':NSTEP,
         'food_xyz':food,'interaction_mode':'online'}
    if cls is not None:
        s['cells'] = CLASSES[cls]; s['passive'] = {'gpas': factor}   # perturb only this class's leak
    return s

def run(sp):
    p = subprocess.run(['xvfb-run','-a','python3',WORKER,json.dumps(sp)],capture_output=True,text=True,timeout=1800)
    for ln in p.stdout.splitlines():
        if ln.startswith('RESULT'): return json.loads(ln[7:])['obs']
    return None

def effdim(e,thr):
    a=np.sort(np.abs(np.asarray(e,float)))[::-1]; a=a[a>0]
    return int(np.searchsorted(np.cumsum(a)/a.sum(),thr)+1) if a.sum() else 0

def hess(food,tag):
    jobs=[(f'{tag}_base',spec(None,1.0,food))]
    for c in CLABELS:
        jobs.append((f'{tag}_{c}+',spec(c,float(np.exp(DELTA)),food)))
        jobs.append((f'{tag}_{c}-',spec(c,float(np.exp(-DELTA)),food)))
    res={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for k,o in ex.map(lambda it:(it[0],run(it[1])),jobs):
            res[k]=o; print(f'{k} {time.time()-t0:.0f}s',flush=True)
    base=res[f'{tag}_base']; bv=np.array([base[k] for k in OBS],float); sc=np.abs(bv)+1e-9
    J=np.zeros((len(OBS),len(CLABELS)))
    for i,c in enumerate(CLABELS):
        op,om=res.get(f'{tag}_{c}+'),res.get(f'{tag}_{c}-')
        if op and om: J[:,i]=((np.array([op[k] for k in OBS])-np.array([om[k] for k in OBS]))/(2*DELTA))/sc
    return J,base

t0=time.time()
Jc,bc=hess(FOOD_CLOSE,'chemo'); Jl,bl=hess(FOOD_FAR,'loco')
Hc=Jc.T@Jc; Hl=Jl.T@Jl; Hu=Hc+Hl
wc=np.linalg.eigvalsh(Hc); wl=np.linalg.eigvalsh(Hl); wu=np.linalg.eigvalsh(Hu)
_,Vc=np.linalg.eigh(Hc); _,Vl=np.linalg.eigh(Hl)
ov=float(np.linalg.norm(Vc[:,-3:].T@Vl[:,-3:])/np.sqrt(3))
# which classes are stiff in each behaviour (top eigvec loadings)
def top_classes(V,n=4): return [CLABELS[i] for i in np.argsort(np.abs(V[:,-1]))[::-1][:n]]
out={'experiment':'B6b_worm_cell-class_union (BAAIWorm chemotaxis vs locomotion)',
  'classes':CLABELS,'delta':DELTA,
  'eff_dim_chemotaxis_90_99':[effdim(wc,.90),effdim(wc,.99)],
  'eff_dim_locomotion_90_99':[effdim(wl,.90),effdim(wl,.99)],
  'eff_dim_UNION_90_99':[effdim(wu,.90),effdim(wu,.99)],
  'stiff_subspace_overlap_top3_cos':ov,
  'stiffest_classes_chemotaxis':top_classes(Vc),'stiffest_classes_locomotion':top_classes(Vl),
  'eig_chemo':[float(x) for x in np.sort(wc)[::-1][:6]],
  'eig_loco':[float(x) for x in np.sort(wl)[::-1][:6]],
  'eig_union':[float(x) for x in np.sort(wu)[::-1][:6]],
  'base_net_disp':{'chemotaxis':float(bc['net_disp']),'locomotion':float(bl['net_disp'])},
  'elapsed_s':round(time.time()-t0,1)}
json.dump(out,open('/root/paper2_B6b_cellunion.json','w'),indent=2)
print(f"\nchemo {out['eff_dim_chemotaxis_90_99']} | loco {out['eff_dim_locomotion_90_99']} | UNION {out['eff_dim_UNION_90_99']} (of {len(CLABELS)})")
print(f"stiff chemo classes: {out['stiffest_classes_chemotaxis']} | loco: {out['stiffest_classes_locomotion']} | overlap cos={ov:.3f}")
print('SAVED /root/paper2_B6b_cellunion.json',flush=True)
