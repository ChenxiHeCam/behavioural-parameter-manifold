"""Isolated NEURON processes, with complete gain/state reset each rollout.

Caches output cell references and records the same float32/subsampled voltages
as the audited worker. State reuse must pass fresh-process parity tests first.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import json,pickle,sys,time
from pathlib import Path
import numpy as np

CHANNELS=['cca1','egl19','egl2','egl36','irk','kcnl','kqt3','kvs1','nca','shk1','shl1','slo1_egl19','slo1_unc2','slo2_egl19','slo2_unc2','unc2']
MODEL=None

class Circuit:
    def __init__(self):
        root=Path(os.environ['BAAI_ROOT']);sys.path.insert(0,str(root));sys.path.insert(0,str(root/'eworm/ghost_in_mesh_sim'));os.chdir(root)
        from eworm.network import transform,detailed_circuit
        from neuron import h
        self.h=h
        cfg=json.loads((root/'eworm/network/config.json').read_text())
        cfg['dir_info']['mechanism_dir']=os.environ['BAAI_MECHANISM_DIR']
        group=root/'eworm/ghost_in_mesh_sim/data/tuned/video_offline'
        with (group/'video_offline_abscircuit.pkl').open('rb') as f:abstract=pickle.load(f)
        self.circuit=transform.abstract2point(abstract,cfg)
        self.inputs=['AWAL','AWAR','AWCL','AWCR','ASKL','ASKR','ALNL','ALNR','PLML','PHAL','PHAR','URYDL','URYDR','URYVL','URYVR']
        names=['RIML','RIMR','RMEL','RMER','RMED','RMEV','RMDDL','RMDDR','RMDL','RMDR','RMDVL','RMDVR','RIVL','RIVR','SMDDL','SMDDR','SMDVL','SMDVR','SMBDL','SMBDR','SMBVL','SMBVR']
        names += [f'{prefix}{i:02d}' for prefix,n in [('DA',9),('DB',7),('DD',6),('VA',12),('VB',11),('VD',13)] for i in range(1,n+1)]
        self.outputs=names
        self.voltages=[self.circuit.cell(cell_name=n).hoc_obj.Soma(.5)._ref_v for n in names]
        self.circuit.input_connections=[]
        for name in self.inputs:self.circuit.add_connection(detailed_circuit.Connection(None,self.circuit.cell(cell_name=name).segments[0],'syn',10))
        self.input_synapses=[c.hoc_obj for c in self.circuit.input_connections]
        self.addresses={}
        for channel in CHANNELS:
            variable='gb'+channel.split('_',1)[0] if channel.startswith(('slo1_','slo2_')) else 'gb'+channel
            addresses=[]
            for sec in h.allsec():
                if sec.has_membrane(channel):
                    for seg in sec:
                        mechanism=getattr(seg,channel)
                        addresses.append((mechanism,variable,float(getattr(mechanism,variable))))
            if not addresses:raise ValueError('Missing mechanism '+channel)
            self.addresses[channel]=addresses

    def run(self,request):
        began=time.time();stim=request['stimulus'];gains=request['log_gains']
        if set(gains)!=set(CHANNELS):raise ValueError('Every log gain must be explicit')
        for channel in CHANNELS:
            factor=float(np.exp(gains[channel]))
            if not np.isfinite(factor) or factor<=0:raise ValueError('Invalid gain')
            for mechanism,variable,original in self.addresses[channel]:setattr(mechanism,variable,original*factor)
        dt=5/3;tstop=1500.;n=int(tstop/dt);times=np.arange(n)*dt
        inputs=np.full((15,n),float(stim['off_mV']),dtype=float)
        for component in stim['components']:
            begin=float(component['start_ms']);end=float(component['end_ms']);active=(times>=begin)&(times<end)
            wave=np.full(n,float(stim['off_mV']))
            if component['shape']=='step':wave[active]=float(component['on_mV'])
            elif component['shape']=='ramp':wave[active]=float(stim['off_mV'])+(float(component['on_mV'])-float(stim['off_mV']))*(times[active]-begin)/(end-begin)
            else:raise ValueError('Invalid stimulus shape')
            for cell in component['cells']:inputs[self.inputs.index(cell)]=wave
        h=self.h;h.dt=dt;h.tstop=tstop;h.secondorder=0
        for synapse in self.input_synapses:synapse.vpre=-65
        h.finitialize(-65)
        stride=max(1,n//120);record=np.empty((80,len(range(0,n,stride))),dtype=np.float32);column=0
        for step in range(n):
            for i,synapse in enumerate(self.input_synapses):synapse.vpre=float(inputs[i,step])
            h.fadvance()
            if step%stride==0:
                record[:,column]=[v[0] for v in self.voltages];column+=1
        if not np.isfinite(record).all():raise ValueError('Nonfinite response')
        return record.astype(float),{'model_sec':time.time()-began,'section_count':sum(1 for _ in h.allsec()),'gain_segments_touched':{c:len(v) for c,v in self.addresses.items()}}

def initialize():
    global MODEL
    MODEL=Circuit()

def evaluate(request):
    if MODEL is None:initialize()
    return MODEL.run(request)
