"""Biological validation of COMPLEMENTARY TILING in real freely-moving C. elegans.
Atanas & Flavell 2023 (Cell) CePNEM encoding atlas: for each recording/time-range, which neurons
encode each behaviour (velocity v, head-curvature th, feeding P). The model predicts different
behaviours constrain COMPLEMENTARY (partially-overlapping, distinct) subspaces (cross-behaviour
cos 0.54). Test: are the real neural encoding sets of different behaviours partially overlapping AND
significantly more segregated than random sets of the same sizes?
Run: python _atanas_tiling.py"""
import h5py, numpy as np
from itertools import combinations
np.random.seed(0)
F='REPRODUCIBLE/paper2_real_experiments/atanas/neuron_categorization.h5'

def behav_of(grp):
    ks=set(grp.keys())
    if {'dorsal','ventral'} & ks: return 'head_curve'
    if {'fwd','rev'} & ks and 'dorsal' not in ks: return 'velocity'
    if {'act','inh'} & ks: return 'feeding'
    return None

pairs={}; sizes=[]; ntot=[]
with h5py.File(F,'r') as f:
    for rec in f['data'].keys():
        for rng in f['data'][rec].keys():
            node=f['data'][rec][rng]
            if not isinstance(node,h5py.Group): continue
            sets={}; alln=set()
            for k in node.keys():
                o=node[k]
                if isinstance(o,h5py.Group):
                    b=behav_of(o)
                    if b and 'all' in o: sets[b]=set(np.array(o['all']).ravel().tolist())
                elif k=='all':
                    alln=set(np.array(o).ravel().tolist())
            if len(sets)<2: continue
            N=max(len(alln), max((max(s) for s in sets.values() if s), default=0))  # neuron count for null
            if N<5: continue
            for b in sets: sizes.append((b,len(sets[b])))
            ntot.append(N)
            for b1,b2 in combinations(sorted(sets),2):
                s1,s2=sets[b1],sets[b2]
                if not s1 or not s2: continue
                inter=len(s1&s2); uni=len(s1|s2)
                jac=inter/uni; cos=inter/np.sqrt(len(s1)*len(s2))
                # random null: two random subsets of N neurons with sizes |s1|,|s2|
                nj=[]; nc=[]
                for _ in range(2000):
                    r1=set(np.random.choice(N,min(len(s1),N),replace=False))
                    r2=set(np.random.choice(N,min(len(s2),N),replace=False))
                    i=len(r1&r2); nj.append(i/len(r1|r2)); nc.append(i/np.sqrt(len(r1)*len(r2)))
                pairs.setdefault(f'{b1}|{b2}',[]).append((jac,cos,np.mean(nj),np.mean(nc)))

print("=== complementary tiling in real C. elegans neural encoding (Atanas 2023) ===")
allreal=[]; allnull=[]
for k,v in sorted(pairs.items()):
    a=np.array(v)  # cols: jac, cos, null_jac, null_cos
    print(f"{k:24s} n={len(a):3d} | real cos {a[:,1].mean():.3f}  null cos {a[:,3].mean():.3f} | real jac {a[:,0].mean():.3f} null jac {a[:,2].mean():.3f}")
    allreal.append(a[:,1].mean()); allnull.append(a[:,3].mean())
# overall: real overlap vs null, and vs the model's cross-behaviour cos 0.54
A=np.vstack([np.array(v) for v in pairs.values()])
from scipy import stats
W,p=stats.wilcoxon(A[:,1],A[:,3],alternative='less')   # real cos < null cos => behaviours segregated beyond chance
out={'experiment':'Atanas2023_tiling_validation','n_pairs':int(A.shape[0]),
 'mean_real_cos':float(A[:,1].mean()),'mean_null_cos':float(A[:,3].mean()),
 'mean_real_jaccard':float(A[:,0].mean()),'mean_null_jaccard':float(A[:,2].mean()),
 'wilcoxon_real_less_than_null_p':float(p),'model_cross_behaviour_cos':0.54,
 'per_pair':{k:{'real_cos':float(np.array(v)[:,1].mean()),'null_cos':float(np.array(v)[:,3].mean())} for k,v in pairs.items()}}
import json; json.dump(out,open('NEXT_PAPER_manifold_subspace/paper2_ATANAS_tiling.json','w'),indent=2)
print(f"\nOVERALL: real cross-behaviour encoding overlap cos={A[:,1].mean():.3f} vs random null {A[:,3].mean():.3f}")
print(f"  Wilcoxon real<null p={p:.2e}  (real<null => behaviours recruit COMPLEMENTARY/segregated ensembles)")
print(f"  model cross-behaviour stiff-subspace cos=0.54 (partial overlap) -- compare to real {A[:,1].mean():.2f}")
print("SAVED paper2_ATANAS_tiling.json")
