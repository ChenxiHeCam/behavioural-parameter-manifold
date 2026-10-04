"""Diagnose normalization and duration effects using the same recovered J."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json
import numpy as np

out=Path(os.environ['PAPER2_OUTPUT_DIR'])
with np.load(out/'E42_motor_voltage_matrices.npz') as z:
    base=z['baseline']; scale=z['output_scale']; J=z['J']
raw=(J.reshape(base.shape+(J.shape[1],))*scale[:,:,None])
diagnostics={}
protocols=[('temporal_SD_per_neuron_2000ms',120,base.std(1).clip(1e-3)[:,None]),
           ('across_neuron_SD_per_time_2000ms',120,base.std(0).clip(1e-3)[None,:]),
           ('global_SD_2000ms',120,np.array([[base.std()]])),
           ('temporal_SD_per_neuron_first1500ms',90,base[:,:90].std(1).clip(1e-3)[:,None])]
for label,n,sc in protocols:
    A=(raw[:,:n,:]/sc[:,:n,None]).reshape(-1,J.shape[1])
    eig=np.maximum(np.linalg.eigvalsh(A.T@A)[::-1],0)
    c=np.cumsum(eig)/eig.sum()
    diagnostics[label]={'d90':int(np.searchsorted(c,.9)+1),'d99':int(np.searchsorted(c,.99)+1),
                       'participation_ratio':float(eig.sum()**2/(eig**2).sum()),
                       'n_observables':A.shape[0]}
    print(label,diagnostics[label],flush=True)
(out/'E42_normalization_duration_diagnostic.json').write_text(json.dumps(diagnostics,indent=2))
