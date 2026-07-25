
import os, sys, time, traceback
import numpy as np
sys.path.insert(0, "/root/BAAIWorm-main/build_headless/build")
sys.path.insert(0, "/root/BAAIWorm-main")
from recovery.simulation.closed_loop_simulator import ClosedLoopSimulator
from recovery.utils.io_utils import load_pickle
PROJECT_ROOT="/root/BAAIWorm-main"
BUILD_DIR="/root/BAAIWorm-main/build_headless/build"
ABS=os.path.join(PROJECT_ROOT,"eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_abscircuit.pkl")
WOUT=os.path.join(PROJECT_ROOT,"eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_wout.pkl")
FOOD=np.array([1.8275,-0.0276,-0.3082],dtype=np.float32)
CKPT="/root/BAAIWorm_clone_ready/BAAIWorm-main/recovery/output/phase2/full5_merge_sota_combo_ionpass_T8/result.pkl"
res=load_pickle(CKPT)
weights=res.get("recovered_weights") or {}
ptypes=res.get("param_type")
if isinstance(ptypes,str): ptypes=ptypes.replace(",","+").split("+")
elif not ptypes: ptypes=list(weights.keys())
N=2000
sim=ClosedLoopSimulator(PROJECT_ROOT,BUILD_DIR,ABS,WOUT,n_init_steps=30,n_sim_steps=N,food_location=FOOD)
t0=time.time()
traj=sim.run_with_custom_weights(weights,ptypes,n_steps=N)
dt=time.time()-t0
arr=np.stack([np.asarray(traj["rel_x"]),np.asarray(traj["rel_y"]),np.asarray(traj["rel_z"])],axis=-1)
out=f"/root/baai_long_sota_cond50_step{N}.npy"
np.save(out,arr)
print("DONE",arr.shape,"dt=",dt,"->",out,flush=True)
