'''One REAL modWorm coupled neural+body rollout under per-mechanism scalings.
argv1 JSON: {gap,syn,leak,Cm,rise,fall,B, n_steps}. Prints RESULT {obs:{...}, sec}.
obs = world-frame locomotion from body centroid trajectory.'''
import sys,types,re,json,time,os
os.chdir('/root/autodl-tmp/modWorm_fresh')
jm=types.ModuleType('julia'); jm.Main=types.SimpleNamespace(eval=lambda *a,**k:None); jm.api=types.ModuleType('julia.api')
sys.modules['julia']=jm; sys.modules['julia.api']=jm.api
sys.path.insert(0,'.')
import numpy as np
from modWorm import network_params as nps, predefined_classes_nv as pnv, predefined_classes_mb as pmb
from modWorm import proprioception_simulation as psim, utils
sp=json.loads(sys.argv[1]); NSTEP=int(sp.get('n_steps',250))
CE=nps.CE
# apply multiplicative scalings on REAL biophysical params
CE.gap_conductances = CE.gap_conductances * sp.get('gap',1.0)
CE.syn_conductances = CE.syn_conductances * sp.get('syn',1.0)
CE.leak_conductances = CE.leak_conductances * sp.get('leak',1.0)
CE.cell_caps = CE.cell_caps * sp.get('Cm',1.0)
CE.synaptic_rise_tau = CE.synaptic_rise_tau * sp.get('rise',1.0)
CE.synaptic_fall_tau = CE.synaptic_fall_tau * sp.get('fall',1.0)
CE.B = CE.B * sp.get('B',1.0)
gap=np.load('/root/autodl-tmp/mw_conn_gap_c302.npy'); syn=np.load('/root/autodl-tmp/mw_conn_syn_c302.npy')
mm=np.load('/root/autodl-tmp/mw_muscle_map96.npy')
nv=pnv.CelegansWorm_NervousSystem_PPC(gap,syn); mb=pmb.CelegansWorm_MuscleBody_PPC(mm)
stim=np.load('/root/autodl-tmp/modWorm_fresh/modWorm/presets_input/input_mat_gentle_post_touch.npy')
t0=time.time()
sol=psim.run_network(nv,mb,stim[:NSTEP])
x=np.asarray(sol['x_solution']); y=np.asarray(sol['y_solution'])
cx=x.mean(1); cy=y.mean(1); dx=np.diff(cx); dy=np.diff(cy)
speed=np.sqrt(dx*dx+dy*dy)
heading=np.arctan2(dy,dx); dh=np.arctan2(np.sin(np.diff(heading)),np.cos(np.diff(heading)))
obs={'net_disp':float(np.sqrt((cx[-1]-cx[0])**2+(cy[-1]-cy[0])**2)),
     'path_len':float(speed.sum()),'mean_speed':float(speed.mean()),
     'speed_std':float(speed.std()),'rms_heading':float(np.sqrt((dh**2).mean())) if dh.size else 0.0,
     'final_curve':float(np.abs(dh).sum()) if dh.size else 0.0}
print('RESULT '+json.dumps({'obs':obs,'sec':round(time.time()-t0,1),'T':int(x.shape[0])}))
