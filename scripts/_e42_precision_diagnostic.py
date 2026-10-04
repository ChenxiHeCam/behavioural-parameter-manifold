"""Check historical float32 arithmetic and alternate output normalization."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np

out = Path(os.environ['PAPER2_OUTPUT_DIR'])
with np.load(out/'responses/connection_-001_+0.npz') as z:
    baseline = z['voltage']
base32 = baseline.astype(np.float32)
sc32 = base32.std(1).clip(1e-3)[:,None]
sc64 = baseline.std(1).clip(1e-3)[:,None]
columns = []
for i in range(3076):
    with np.load(out/f'responses/connection_{i:04d}_+1.npz') as z:
        p = z['voltage'].astype(np.float32)
    with np.load(out/f'responses/connection_{i:04d}_-1.npz') as z:
        q = z['voltage'].astype(np.float32)
    columns.append(((p-q)/(2*.5)/sc32).ravel())
J = np.array(columns,dtype=np.float64).T
G = J.T@J
eig = np.maximum(np.linalg.eigvalsh(G)[::-1],0)
cum = np.cumsum(eig)/eig.sum()
diagnostic = {'description':'Exact deposited E42 arithmetic on recovered raw responses',
              'd90':int(np.searchsorted(cum,.90)+1),'d99':int(np.searchsorted(cum,.99)+1),
              'participation_ratio':float(eig.sum()**2/(eig**2).sum()),
              'baseline_scale_max_relative_change_float32_vs_float64':float(np.max(np.abs(sc32-sc64)/sc64)),
              'baseline_std_float32':float(base32.std()),'baseline_std_float64':float(baseline.std()),
              'elasticity_percentiles':{str(p):float(np.percentile(np.linalg.norm(J,axis=0),p)) for p in (1,25,50,75,99)},
              'eigenvalues':eig.tolist()}
(out/'E42_precision_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
np.savez_compressed(out/'E42_deposited_precision_matrices.npz',J=J,G=G,eigenvalues=eig)
print({k:v for k,v in diagnostic.items() if k!='eigenvalues'},flush=True)
