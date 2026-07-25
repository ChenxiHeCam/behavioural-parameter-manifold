import sys,types,re,time
jm=types.ModuleType('julia'); jm.Main=types.SimpleNamespace(eval=lambda *a,**k:None); jm.api=types.ModuleType('julia.api')
sys.modules['julia']=jm; sys.modules['julia.api']=jm.api
sys.path.insert(0,'/root/autodl-tmp/modWorm_fresh')
import numpy as np, pandas as pd
from modWorm import utils, predefined_classes_nv as pnv, predefined_classes_mb as pmb
from modWorm import proprioception_simulation as psim, sys_paths
nn=list(utils.neuron_names); idx={n:i for i,n in enumerate(nn)}
def norm(s):
    s=str(s); m=re.match(r'^([A-Za-z]+)(\d+)$',s)
    return (m.group(1)+m.group(2).zfill(2)) if m else s
gap=np.load('/root/autodl-tmp/mw_conn_gap_c302.npy'); syn=np.load('/root/autodl-tmp/mw_conn_syn_c302.npy')
# muscle map (95,279) replicating construct_muscle_map_Hall, from c302
seg_names=['MDL','MDR','MVL','MVR']; segs=[]
for k in range(1,25):
    for q in seg_names: segs.append(q+str(k).zfill(2))
# keep MVL24 as zero row -> 96 rows (fix 24v23)
dfm=pd.read_excel('/root/autodl-tmp/owconn.csv',sheet_name='NeuronsToMuscle')
mm=np.zeros((96,279))
for neu,mus,num in zip(dfm['Neuron'],dfm['Muscle'],dfm['Number of Connections']):
    n=norm(neu); mus=str(mus)
    if mus in segs and n in idx: mm[segs.index(mus), idx[n]]+=num
print('muscle map nnz',int((mm>0).sum()),'rows-with-input',int((mm.sum(1)>0).sum()),'/96',flush=True)
# build nervous + muscle body (pure-python PPC)
nv=pnv.CelegansWorm_NervousSystem_PPC(gap, syn)
mb=pmb.CelegansWorm_MuscleBody_PPC(mm)
print('classes built. neuron_num',nv.network_Size,flush=True)
# input: bundled gentle posterior touch
stim=np.load('/root/autodl-tmp/modWorm_fresh/modWorm/presets_input/input_mat_gentle_post_touch.npy')
print('stim shape',stim.shape,flush=True)
# SHORT rollout: first 200 steps (2s) to time
t0=time.time()
sol=psim.run_network(nv, mb, stim[:200])
dt=time.time()-t0
print('RUN OK keys',list(sol.keys()),'time_200steps=%.1fs'%dt,flush=True)
x=np.asarray(sol['x_solution']); y=np.asarray(sol['y_solution'])
print('traj x',x.shape,'net_disp',float(np.sqrt((x[-1,0]-x[0,0])**2+(y[-1,0]-y[0,0])**2)),flush=True)
