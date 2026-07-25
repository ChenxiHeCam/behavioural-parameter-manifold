'''DETERMINISTIC BAAIWorm world-frame rollout. spec(JSON argv1):
 {ion:{gbkey:f}, passive:{k:f}, syn:f, gj:f, wout:f, wfile:path, n_steps:120, food_idx:0}
 wfile = npz with syn_mult(1992),gj_mult(1084) per-connection multipliers (overrides scalar syn/gj).
Prints: RESULT {obs:{net_disp,path_len,mean_speed,speed_std,rms_heading,fwd_world}, sec, T}'''
import os,sys,json,copy,time,pickle
import numpy as np
for _a,_t in [('float',float),('int',int),('bool',bool),('object',object)]:
    if not hasattr(np,_a): setattr(np,_a,_t)
REPO='/root/BAAIWorm_fresh'; GHOST=os.path.join(REPO,'eworm','ghost_in_mesh_sim')
sys.path[:0]=[os.path.join(REPO,'Metaworm','interact'),REPO,GHOST]
import abstract_circuit
GDIR=os.path.join(GHOST,'data','tuned','video_offline')
ABS_PKL=GDIR+'/video_offline_abscircuit.pkl'; WOUT_PKL=GDIR+'/video_offline_wout.pkl'
FOODS=[np.array([1.8275,-0.0276,-0.3082],dtype=np.float32),np.array([1.5,0.2,-0.5],dtype=np.float32)]
spec=json.loads(sys.argv[1])
ION=spec.get('ion',{}); PAS=spec.get('passive',{}); SYN=spec.get('syn',1.0); GJ=spec.get('gj',1.0); WOUT=spec.get('wout',1.0)
WFILE=spec.get('wfile',None); NSTEP=int(spec.get('n_steps',120))
FOOD=np.array(spec['food_xyz'],dtype=np.float32) if 'food_xyz' in spec else FOODS[int(spec.get('food_idx',0))]
CELLS=spec.get('cells',None)  # list of cell-name prefixes to perturb; None=all (per-neuron-class)
def world_obs(world,dt=1.0):
    w=np.asarray(world)
    if w.ndim!=2 or w.shape[0]<5: return None
    cx=w[:,0]; cz=w[:,2]; dx=np.diff(cx); dz=np.diff(cz)
    speed=np.sqrt(dx*dx+dz*dz)/dt
    heading=np.arctan2(dz,dx); dh=np.arctan2(np.sin(np.diff(heading)),np.cos(np.diff(heading)))
    net=float(np.sqrt((cx[-1]-cx[0])**2+(cz[-1]-cz[0])**2))
    return {'net_disp':net,'path_len':float(speed.sum()),'mean_speed':float(speed.mean()),
            'speed_std':float(speed.std()),'rms_heading':float(np.sqrt((dh**2).mean())),
            'fwd_world':float((speed>np.percentile(speed,25)).mean())}
from eworm.network.detailed_circuit import Cell
_orig=Cell.setup_biophysics
def wrapped(self,cell_param):
    if CELLS is not None and not any(str(getattr(self,'name','')).startswith(c) for c in CELLS):
        return _orig(self,cell_param)   # cell not in target class -> no perturbation
    cp=copy.deepcopy(cell_param)
    for cat in cp:
        if not isinstance(cp[cat],dict): continue
        for k in list(cp[cat].keys()):
            v=cp[cat][k]
            if not isinstance(v,(int,float)): continue
            if k.startswith('gb') and k in ION: cp[cat][k]=v*ION[k]
            elif k in PAS: cp[cat][k]=v*PAS[k]
    return _orig(self,cp)
Cell.setup_biophysics=wrapped
def scaled_abscircuit():
    c=pickle.load(open(ABS_PKL,'rb'))
    if WFILE:
        z=np.load(WFILE); sm=z['syn_mult']; gm=z['gj_mult']; si=gi=0
        for conn in c.connections:
            if conn.category=='syn': conn.weight=conn.weight*float(sm[si]); si+=1
            elif conn.category=='gj': conn.weight=conn.weight*float(gm[gi]); gi+=1
    else:
        for conn in c.connections:
            if conn.category=='syn': conn.weight=conn.weight*SYN
            elif conn.category=='gj': conn.weight=conn.weight*GJ
    return c
import eworm.ghost_in_mesh_sim.worm_neural_network as wnn
import interact; interact.init_environment()
abs_c=scaled_abscircuit(); wout=pickle.load(open(WOUT_PKL,'rb'))
wout_s=(np.asarray(wout)*WOUT) if isinstance(wout,(np.ndarray,list)) else wout
_real=pickle.load
def fake(f,*a,**k):
    try: nm=f.name
    except Exception: nm=''
    if nm.endswith('_abscircuit.pkl'): return abs_c
    if nm.endswith('_wout.pkl'): return wout_s
    return _real(f,*a,**k)
wnn.pickle.load=fake
from worm_in_env import WormInEnv
t0=time.time(); env=WormInEnv(food_location=FOOD, target_location=[FOOD])  # target_location drives the chemosensory input (NOT food_location)
env.config['interaction_mode']=spec.get('interaction_mode','online')  # LIVE food-derived chemotaxis (override video_offline default)
ni=min(10,len(env.init_muscle_file))
for _ in range(ni):
    env.step_init(); interact.step_physics()
env.step_count=0; world=[]
for _ in range(NSTEP):
    env.step(); interact.step_physics()
    hl=interact.get_head_location(0); world.append(np.asarray(interact.worm_local_to_world(0,hl)).copy())
obs=world_obs(world)
print('RESULT '+json.dumps({'obs':obs,'sec':round(time.time()-t0,1),'T':len(world)}))
