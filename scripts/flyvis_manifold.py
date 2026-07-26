'''flyvis fly connectome manifold/sloppiness probe: random perturbations of the 734
free params -> response dispersion. Shows params vary while response stays clustered.'''
import json,time,numpy as np,torch
torch.set_num_threads(8)
import flyvis
from flyvis import Network
t0=time.time(); net=Network()
free=[p for n,p in net.named_parameters() if p.requires_grad]
orig=[p.detach().clone() for p in free]
n_in=721;T=20;dt=1/50; xx=np.arange(n_in)
mov=np.stack([0.5+0.5*np.sin(2*np.pi*(xx/40.0-f*0.1)) for f in range(T)])[None]
movie=torch.tensor(mov,dtype=torch.float32).unsqueeze(2)
def sim():
    with torch.no_grad(): return net.simulate(movie,dt)[0,-5:].mean(0).detach().numpy()
a0=sim(); scale=np.abs(a0)+1e-6
N=30; SIG=0.20; rng=np.random.RandomState(7); bd=[]; pr=[]
for k in range(N):
    rel=[]
    for p,o in zip(free,orig):
        f=torch.tensor(np.exp(rng.normal(0,SIG,size=tuple(p.shape))),dtype=p.dtype)
        p.data.copy_(o*f); rel.append(float(np.mean(np.abs(f.numpy()-1))))
    a=sim()
    bd.append(float(np.linalg.norm((a-a0)/scale)/np.sqrt(a0.size)))
    pr.append(float(np.mean(rel)))
    for p,o in zip(free,orig): p.data.copy_(o)
bd=np.array(bd); pr=np.array(pr)
out={'sim':'flyvis Drosophila connectome model','probe':'manifold/sloppiness sample','N':N,'sigma':SIG,
 'param_relerr_mean':float(pr.mean()),'response_dist_mean':float(bd.mean()),'response_over_param_ratio':float(bd.mean()/max(pr.mean(),1e-9)),
 'elapsed_sec':round(time.time()-t0)}
json.dump(out,open('/root/autodl-tmp/flyvis_manifold.json','w'),indent=2)
print('=== flyvis manifold DONE ===',flush=True)
print('param relerr=%.3f response dist=%.4f response/param=%.3f'%(pr.mean(),bd.mean(),out['response_over_param_ratio']),flush=True)
