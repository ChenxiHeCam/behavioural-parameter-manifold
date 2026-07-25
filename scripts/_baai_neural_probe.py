import os, sys, time, pickle
import numpy as np
sys.path[:0]=['/root/BAAIWorm/Metaworm/interact','/root/BAAIWorm','/root/BAAIWorm/eworm/ghost_in_mesh_sim']
from neuron import h, load_mechanisms
load_mechanisms('/root/BAAIWorm/Metaworm/interact/neural_model/components/mechanism/modfile')
from eworm.utils import func
from eworm.network import transform, detailed_circuit

BASE='/root/BAAIWorm/eworm/ghost_in_mesh_sim/data/tuned/video_offline/'
config = func.load_json(BASE+'video_offline_config.json')['config']
abs_circuit = pickle.load(open(BASE+'video_offline_abscircuit.pkl','rb'))
print('abscircuit loaded; building detailed circuit ...', flush=True)
t0=time.time()
circuit = transform.abstract2detailed(abs_circuit, config, load_hoc=True, rec_voltage=True)
print('built in %.1fs' % (time.time()-t0), flush=True)

chemo=["AWAL","AWAR","AWCL","AWCR"]
circuit.input_connections=[]
for cn in chemo:
    circuit.add_connection(detailed_circuit.Connection(None, circuit.cell(cell_name=cn).segments[0], 'syn', 10))
motor=["DA01","DA02","DA03","DB01","DB02","DB03","VA01","VA02","VA03","VB01","VB02","VB03","DD01","DD02","VD01","VD02"]
sim_config={"dt":5/3,"tstop":3000,"v_init":-65,"secondorder":0}
L=int(sim_config['tstop']/sim_config['dt'])
t=np.arange(L)
inp=np.tile((-30+30*np.sin(2*np.pi*t/120)).astype(np.float32),(len(chemo),1))   # fluctuating chemosensory drive
print('simulating (%d steps)...'%L, flush=True)
t0=time.time()
out=circuit.simulation(sim_config, inp, chemo, motor)
print('SIM OK: out shape %s, %.1fs, out std %.3f, out range [%.1f,%.1f]' % (out.shape, time.time()-t0, out.std(), out.min(), out.max()), flush=True)
