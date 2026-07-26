'''flyvis fly connectome B1 via FINITE DIFFERENCE (simulate runs no-grad).
Perturb each of the 65 named cell-type biases +/-delta, re-simulate, measure
response change -> per-cell-type elasticity Jacobian -> J^T J eff-dim + stiff/sloppy.
Also 3-group summary (bias / time_const / syn_strength).'''
import json,time,numpy as np,torch
torch.set_num_threads(8)
import flyvis
from flyvis import Network
t0=time.time(); net=Network()
free={n:p for n,p in net.named_parameters() if p.requires_grad}
# cell-type names for the 65 node params
try:
    ct=[str(x) for x in net.connectome.nodes.type[:]]
    import numpy as _np
    uniq=list(dict.fromkeys(ct))   # ordered unique cell types
except Exception: uniq=[f'ct{i}' for i in range(65)]
n_in=721; T=20; dt=1/50
xx=np.arange(n_in)
mov=np.stack([0.5+0.5*np.sin(2*np.pi*(xx/40.0 - f*0.1)) for f in range(T)])[None]
movie=torch.tensor(mov,dtype=torch.float32).unsqueeze(2)
def sim():
    with torch.no_grad():
        act=net.simulate(movie, dt)
        return act[0,-5:].mean(0).detach().numpy()
ts=time.time(); a0=sim(); print('1 simulate %.1fs, response dim %d'%(time.time()-ts,a0.size),flush=True)
g=64; idx=np.linspace(0,a0.size,g+1).astype(int)
def readout(a): return np.array([a[idx[k]:idx[k+1]].mean() for k in range(g)])
r0=readout(a0); scale=np.abs(r0)+1e-6
DELTA=0.2
bias=free['nodes_bias']; nct=bias.numel()
names=uniq[:nct] if len(uniq)>=nct else [f'ct{i}' for i in range(nct)]
J=np.zeros((g,nct))
orig=bias.detach().clone()
for i in range(nct):
    d=DELTA*max(abs(float(orig[i])),0.05)
    bias.data[i]=orig[i]+d; rp=readout(sim())
    bias.data[i]=orig[i]-d; rm=readout(sim())
    bias.data[i]=orig[i]
    J[:,i]=((rp-rm)/scale)/(2*d)
    if i%15==0: print('ct',i,'el',round(time.time()-t0),flush=True)
H=J.T@J; ev=np.sort(np.linalg.eigvalsh(H))[::-1]; ev=np.maximum(ev,0)
cs=np.cumsum(ev)/max(ev.sum(),1e-30); eff90=int(np.searchsorted(cs,0.9)+1); eff99=int(np.searchsorted(cs,0.99)+1)
pr=float((ev.sum()**2)/np.maximum((ev**2).sum(),1e-30))
stiff={names[i]:float(np.linalg.norm(J[:,i])) for i in range(nct)}
order=sorted(stiff,key=lambda k:-stiff[k])
out={'sim':'flyvis Drosophila optic-lobe connectome (Lappalainen 2024)','probe':'per-cell-type bias FD elasticity Hessian',
 'n_cell_types':nct,'delta':DELTA,'readout_dim':g,'eff_dim_90':eff90,'eff_dim_99':eff99,'participation_ratio':pr,
 'top_eig':ev[:10].tolist(),'spectral_span_orders':float(np.log10(ev[0]/ev[ev>0][-1])) if (ev>0).any() else None,
 'rank_stiff_to_sloppy':[(names[i] if i<len(names) else f'ct{i}', round(stiff[order[j]],4)) for j,i in enumerate([names.index(o) if o in names else 0 for o in order])][:20],
 'per_celltype_stiffness':stiff,'elapsed_sec':round(time.time()-t0)}
json.dump(out,open('/root/autodl-tmp/flyvis_b1.json','w'),indent=2)
print('=== flyvis B1 (FD) DONE ===',flush=True)
print('eff_dim 90%%=%d 99%%=%d /%d PR=%.2f'%(eff90,eff99,nct,pr),flush=True)
print('STIFF cell types:',[(o,round(stiff[o],3)) for o in order[:6]],flush=True)
print('SLOPPY:',[(o,round(stiff[o],3)) for o in order[-4:]],flush=True)
