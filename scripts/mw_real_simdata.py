'''Generate REAL modWorm simulation dataset for NPE/SBC: sample 7-mechanism mults
from prior, run real modWorm, collect 6 behavioural observables. Pool-parallel
(each proc imports modWorm once).'''
import os,sys,json,time
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from multiprocessing import Pool
N=int(os.environ.get('N','1200')); NSTEP=int(os.environ.get('NSTEP','200')); PROCS=int(os.environ.get('PROCS','40'))
MECH=['gap','syn','leak','Cm','rise','fall','B']; OBS=['net_disp','path_len','mean_speed','speed_std','rms_heading','final_curve']
SIG=0.30  # prior: lognormal around default (mult=1) in log space sd
rng=np.random.RandomState(2024)
THETAS=np.exp(rng.normal(0,SIG,size=(N,7)))  # (N,7) positive mults
def init():
    global pnv,pmb,psim,nps,GAP,SYN,MM,STIM,D0,copy
    import types,copy as _copy; copy=_copy
    os.chdir('/root/autodl-tmp/modWorm_fresh')
    jm=types.ModuleType('julia'); jm.Main=types.SimpleNamespace(eval=lambda *a,**k:None); jm.api=types.ModuleType('julia.api')
    sys.modules['julia']=jm; sys.modules['julia.api']=jm.api; sys.path.insert(0,'.')
    from modWorm import network_params as _nps, predefined_classes_nv as _pnv, predefined_classes_mb as _pmb, proprioception_simulation as _psim
    pnv,pmb,psim,nps=_pnv,_pmb,_psim,_nps
    GAP=np.load('/root/autodl-tmp/mw_conn_gap_c302.npy'); SYN=np.load('/root/autodl-tmp/mw_conn_syn_c302.npy'); MM=np.load('/root/autodl-tmp/mw_muscle_map96.npy')
    STIM=np.load('/root/autodl-tmp/modWorm_fresh/modWorm/presets_input/input_mat_gentle_post_touch.npy')[:NSTEP]
    D0={k:copy.deepcopy(getattr(nps.CE,a)) for k,a in [('gap','gap_conductances'),('syn','syn_conductances'),('leak','leak_conductances'),('Cm','cell_caps'),('rise','synaptic_rise_tau'),('fall','synaptic_fall_tau'),('B','B')]}
def one(theta):
    try:
        m=dict(zip(MECH,theta))
        nps.CE.gap_conductances=D0['gap']*m['gap']; nps.CE.syn_conductances=D0['syn']*m['syn']
        nps.CE.leak_conductances=D0['leak']*m['leak']; nps.CE.cell_caps=D0['Cm']*m['Cm']
        nps.CE.synaptic_rise_tau=D0['rise']*m['rise']; nps.CE.synaptic_fall_tau=D0['fall']*m['fall']; nps.CE.B=D0['B']*m['B']
        nv=pnv.CelegansWorm_NervousSystem_PPC(GAP,SYN); mb=pmb.CelegansWorm_MuscleBody_PPC(MM)
        sol=psim.run_network(nv,mb,STIM); x=np.asarray(sol['x_solution']); y=np.asarray(sol['y_solution'])
        cx=x.mean(1); cy=y.mean(1); dx=np.diff(cx); dy=np.diff(cy); sp=np.sqrt(dx*dx+dy*dy)
        hd=np.arctan2(dy,dx); dh=np.arctan2(np.sin(np.diff(hd)),np.cos(np.diff(hd)))
        return [float(np.sqrt((cx[-1]-cx[0])**2+(cy[-1]-cy[0])**2)),float(sp.sum()),float(sp.mean()),float(sp.std()),float(np.sqrt((dh**2).mean()) if dh.size else 0),float(np.abs(dh).sum() if dh.size else 0)]
    except Exception:
        return [np.nan]*6
t0=time.time()
with Pool(PROCS,initializer=init) as p:
    X=p.map(one, THETAS, chunksize=2)
X=np.array(X,float)
np.savez('/root/autodl-tmp/mw_real_simdata.npz', theta=THETAS, x=X, mech=MECH, obs_keys=OBS)
ok=int(np.all(np.isfinite(X),1).sum())
print('SIMDATA done N=%d ok=%d %.0fs'%(N,ok,time.time()-t0),flush=True)
