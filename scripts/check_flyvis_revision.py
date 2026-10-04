"""Independently recalculate every retained flyvis spectrum and inspect convergence."""
import json
from pathlib import Path
import hashlib
import numpy as np


def check(path):
    d=json.loads(path.read_text(encoding='utf-8'))
    arr=path.with_suffix('.npz')
    assert hashlib.sha256(arr.read_bytes()).hexdigest()==d['arrays_sha256']
    a=np.load(arr)
    interior=a['interior_mask']
    assert len(d['coordinate_manifest'])==sum(len(x) for x in d['selected_indices'].values())
    assert int(interior.sum())==d['interior_count']
    assert d['parameters_restored_exactly']
    primary=a[f"J_scaled_step_{d['primary_step']:g}"][:64]
    reports={}
    for h,grids in d['spectra_by_step'].items():
        J=a[f'J_scaled_step_{h}']
        np.testing.assert_allclose(J,a[f'J_raw_step_{h}']/a['output_scales'][:,None],rtol=1e-13)
        offset=0
        for g in [64,32,16,8,6]:
            j=J[offset:offset+g]
            for name,mask in [('interior',interior),('including_boundary',np.ones(len(interior),dtype=bool)),
                              ('bias_only',np.array([e['group']=='nodes_bias' for e in d['coordinate_manifest']]))]:
                stored=grids[str(g)][name]
                ev=np.linalg.eigvalsh(j[:,mask]@j[:,mask].T)[::-1]
                ev=np.maximum(ev,0)
                # Output-space eigendecomposition loses absolute precision in
                # small modes when baseline normalization makes the leading
                # mode very large. Use a matrix-scale roundoff bound, not a
                # fixed absolute tolerance unrelated to the matrix magnitude.
                roundoff = 8*np.finfo(float).eps*max(j[:,mask].shape)*max(float(ev[0]),1.)
                np.testing.assert_allclose(stored['eigenvalues'][:len(ev)],ev,rtol=2e-10,atol=roundoff)
                assert len(stored['eigenvalues'])==int(mask.sum())
                assert stored['eff_dim_99']<=min(g,int(mask.sum()))
                mass=ev.sum()
                assert stored['eff_dim_90']==int(np.searchsorted(np.cumsum(ev),.9*mass)+1)
                assert stored['eff_dim_99']==int(np.searchsorted(np.cumsum(ev),.99*mass)+1)
            offset+=g
        j=J[:64]
        cn=np.linalg.norm(primary,axis=0)
        relative=np.linalg.norm(j-primary,axis=0)/np.maximum(cn,1e-300)
        # Flag the derivative differences without silently discarding weak axes.
        fail=relative>.1
        reports[h]={'columns_above_10_percent_difference':int(fail.sum()),
                    'fraction_of_primary_sensitivity_in_these_columns':float((cn[fail]**2).sum()/(cn**2).sum()),
                    'column_indices':np.flatnonzero(fail).tolist()}
    return {'file':path.name,'spectrum_recalculation':'PASS','step_column_report':reports}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'results/revision_20261004'
    out={}
    for state in ['trained','initialized']:
        path=root/f'flyvis_{state}_dimensionless.json'
        if path.exists():out[state]=check(path)
    print(json.dumps(out,indent=2))
    (root/'flyvis_verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
