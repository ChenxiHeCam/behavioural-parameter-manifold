'''REAL modWorm NPE + per-dim SBC + three-measure agreement (replaces surrogate).
Trains amortised posterior on real modWorm sims, runs SBC, and correlates
per-mechanism Hessian stiffness (B1) vs manifold drift vs SBC calibration.'''
import json,numpy as np,torch
torch.manual_seed(0)
d=np.load('/root/autodl-tmp/mw_real_simdata.npz',allow_pickle=True)
theta=d['theta'].astype(float); X=d['x'].astype(float); MECH=[str(m) for m in d['mech']]
ok=np.all(np.isfinite(X),1)&np.all(np.isfinite(theta),1); theta=theta[ok]; X=X[ok]
N=len(theta); print('usable sims',N,flush=True)
logt=np.log(theta)  # ~N(0,0.3) per dim
# standardize observables
Xm=X.mean(0); Xs=X.std(0)+1e-9; Xz=(X-Xm)/Xs
import torch
th=torch.tensor(logt,dtype=torch.float32); xo=torch.tensor(Xz,dtype=torch.float32)
from sbi.inference import NPE
from torch.distributions import Normal, Independent
prior=Independent(Normal(torch.zeros(7),0.15*torch.ones(7)),1)
inf=NPE(prior=prior)
inf.append_simulations(th,xo); dens=inf.train(max_num_epochs=200,show_train_summary=False)
post=inf.build_posterior()
print('NPE trained',flush=True)
# SBC: per-dim rank statistic
M=200; L=300
idx=np.random.RandomState(1).choice(N,M,replace=False)
ranks=np.zeros((M,7))
for j,ii in enumerate(idx):
    xobs=xo[ii]; samp=post.sample((L,),x=xobs,show_progress_bars=False).numpy()
    ranks[j]=(samp<logt[ii]).mean(0)  # fraction below true = rank stat in [0,1]
from scipy import stats
sbc_p=[float(stats.kstest(ranks[:,k],'uniform').pvalue) for k in range(7)]
sbc_pass=[p>0.05 for p in sbc_p]
# manifold drift per mechanism: std of param among behaviourally-close sims
bd=np.linalg.norm(Xz - ((np.zeros(6)-Xm)/Xs),axis=1)  # dist from default-behaviour(0 in raw? use median)
close=bd<np.percentile(bd,25)
drift=[float(np.std(logt[close,k])) for k in range(7)]
# Hessian stiffness from B1
b1=json.load(open('/root/autodl-tmp/modworm_REAL_b1.json'))
stiff=[b1['per_mechanism_elasticity'].get(m,0.0) for m in MECH]
rho_hd,p_hd=stats.spearmanr(stiff,[-x for x in drift])   # stiff vs negative drift
rho_hs,p_hs=stats.spearmanr(stiff,[1-p for p in sbc_p])  # stiff vs SBC-determinacy
res={'sim':'REAL modWorm (c302 connectome)','n_sims':N,'dim':7,'mech':MECH,
 'sbc_pass_count':int(sum(sbc_pass)),'sbc_pass_per_dim':dict(zip(MECH,sbc_pass)),'sbc_ks_p':dict(zip(MECH,sbc_p)),
 'hessian_stiffness':dict(zip(MECH,stiff)),'manifold_drift':dict(zip(MECH,drift)),
 'three_measure_hessian_vs_negdrift_spearman':[float(rho_hd),float(p_hd)],
 'three_measure_hessian_vs_sbc_spearman':[float(rho_hs),float(p_hs)]}
json.dump(res,open('/root/autodl-tmp/modworm_REAL_sbc.json','w'),indent=2)
print('=== REAL modWorm SBC + 3-measure DONE ===',flush=True)
print('SBC pass %d/7'%sum(sbc_pass), dict(zip(MECH,[round(p,3) for p in sbc_p])),flush=True)
print('3-measure Hessian-vs-negdrift rho=%.3f p=%.3f'%(rho_hd,p_hd),flush=True)
