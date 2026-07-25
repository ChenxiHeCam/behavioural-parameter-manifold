"""WELL-POSED worm test (finite, by simulation, no local-Hessian basis):
Does real CaeNDR-covariance-SHAPED conductance variation preserve behaviour MORE than isotropic
variation of the SAME total magnitude? If yes -> real biological co-variation is behaviour-preserving
(the prediction's core), which the local-linear B5 test could not see.
Behaviour = motor-neuron voltage traces from the BAAIWorm open-loop neural sim.
Run: /root/miniconda3/bin/python _baai_wellposed.py"""
import os, sys, time, pickle, json, copy
import numpy as np
sys.path[:0]=['/root/BAAIWorm/Metaworm/interact','/root/BAAIWorm','/root/BAAIWorm/eworm/ghost_in_mesh_sim']
from neuron import h, load_mechanisms
load_mechanisms('/root/BAAIWorm/Metaworm/interact/neural_model/components/mechanism/modfile')
from eworm.utils import func
from eworm.network import transform, detailed_circuit
from eworm.network.detailed_circuit import Cell
np.random.seed(0)

BASE='/root/BAAIWorm/eworm/ghost_in_mesh_sim/data/tuned/video_offline/'
CONFIG=func.load_json(BASE+'video_offline_config.json')['config']
ABS=pickle.load(open(BASE+'video_offline_abscircuit.pkl','rb'))
cov=json.load(open('/root/_cendr_cov14.json')); GENES=cov['genes']; L=np.array(cov['chol'])
tr=np.trace(np.array(cov['cov'])); ISO=np.sqrt(tr/14)        # isotropic std matching total variance

CHANNELS=['gbshl1','gbshk1','gbkvs1','gbegl2','gbegl36','gbkqt3','gbegl19','gbunc2','gbcca1',
 'gbslo1_egl19','gbslo1_unc2','gbslo2_egl19','gbslo2_unc2','gbkcnl','gbnca','gbirk']
CHGENE=['shl-1','shk-1','kvs-1','egl-2','egl-36','kqt-3','egl-19','unc-2','cca-1',
 'slo-1','slo-1','slo-2','slo-2','kcnl-1','nca-2','irk-1']
gidx={g:i for i,g in enumerate(GENES)}
CHEMO=["AWAL","AWAR","AWCL","AWCR"]
MOTOR=["DA01","DA02","DA03","DA04","DA05","DB01","DB02","DB03","DB04","VA01","VA02","VA03","VA04","VA05",
 "VB01","VB02","VB03","VB04","DD01","DD02","DD03","VD01","VD02","VD03"]
SIMC={"dt":5/3,"tstop":3000,"v_init":-65,"secondorder":0}
Ln=int(SIMC['tstop']/SIMC['dt']); t=np.arange(Ln)
INP=np.tile((-30+30*np.sin(2*np.pi*t/120)).astype(np.float32),(len(CHEMO),1))

_orig=Cell.setup_biophysics
def run(gene_pert):
    mult={CHANNELS[i]:float(np.exp(gene_pert[gidx[CHGENE[i]]])) for i in range(len(CHANNELS))} if gene_pert is not None else {}
    def wrapped(self,cell_param):
        cp=copy.deepcopy(cell_param)
        for cat in cp:
            if not isinstance(cp[cat],dict): continue
            for k in list(cp[cat].keys()):
                v=cp[cat][k]
                if isinstance(v,(int,float)) and k in mult: cp[cat][k]=v*mult[k]
        return _orig(self,cp)
    Cell.setup_biophysics=wrapped
    c=transform.abstract2detailed(ABS,CONFIG,load_hoc=True,rec_voltage=True)
    Cell.setup_biophysics=_orig
    c.input_connections=[]
    for cn in CHEMO: c.add_connection(detailed_circuit.Connection(None,c.cell(cell_name=cn).segments[0],'syn',10))
    out=c.simulation(SIMC,INP,CHEMO,MOTOR)
    del c
    return out

base=run(None); bscale=base.std()+1e-9
def bdist(o): return float(np.sqrt(np.mean((o-base)**2))/bscale)

t0=time.time(); N=15
real_b=[]; iso_b=[]
for i in range(N):
    z=np.random.randn(14)
    rb=bdist(run(L@z)); ib=bdist(run(ISO*z))
    real_b.append(rb); iso_b.append(ib)
    print(f"sample {i}: real-shaped bdist {rb:.3f} | isotropic bdist {ib:.3f}  ({time.time()-t0:.0f}s)",flush=True)
real_b=np.array(real_b); iso_b=np.array(iso_b)
from scipy import stats
W,p=stats.wilcoxon(real_b,iso_b,alternative='less')        # predict real < iso
out={'experiment':'BAAIWorm_wellposed_curved_test','n':N,
 'median_real_shaped_bdist':float(np.median(real_b)),'median_isotropic_bdist':float(np.median(iso_b)),
 'real_over_iso_ratio':float(np.median(real_b)/np.median(iso_b)),
 'wilcoxon_real_less_than_iso_p':float(p),
 'real_bdist':real_b.tolist(),'iso_bdist':iso_b.tolist(),
 'mean_offdiag_corr':cov['mean_abs_offdiag_corr'],'per_gene_logstd':cov['mean_diag_std']}
json.dump(out,open('/root/paper2_BAAI_wellposed.json','w'),indent=2)
print(f"\nMEDIAN behavioural change: real-covariance-shaped {np.median(real_b):.3f}  vs  isotropic {np.median(iso_b):.3f}  (ratio {out['real_over_iso_ratio']:.2f})")
print(f"Wilcoxon (real < iso) p={p:.4f}  -> if real<iso, real biological co-variation IS behaviour-preserving")
print("SAVED /root/paper2_BAAI_wellposed.json")
