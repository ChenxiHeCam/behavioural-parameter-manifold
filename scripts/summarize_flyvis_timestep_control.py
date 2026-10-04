"""Compare integration choices on the same physical stimulus and reference W."""
import hashlib
import json
from pathlib import Path
import numpy as np
from flyvis_sensitivity_revision import spectrum, comparison

root=Path(__file__).resolve().parents[1]/'results/revision_20261004'
native=json.loads((root/'flyvis_trained_dimensionless.json').read_text())
baseline=np.load(root/'flyvis_trained_dimensionless.npz')
mask=baseline['interior_mask']
identities=[(e['group'],e['index']) for e in native['coordinate_manifest']]
reference=baseline['J_raw_step_0.01']/baseline['output_scales'][:,None]
out={'status':'posthoc numerical control; native results are preserved',
     'checkpoint_training_time_step_seconds':.02,
     'same_physical_input':{'gray_duration_seconds':1.,'movie_duration_seconds':.4,
                            'original_movie_frames':20,'frame_hold_seconds':.02,
                            'final_average_seconds':.1},
     'parameter_coordinate_scope':'same703 native-interior coordinates; inclusive734 separate',
     'common_weight':'native .02 trained baseline absolute bin means plus1e-6',
     'solver_floor':'official dynamics divide by max(time_constant, dt), so decreasing dt also releases51 native clipped taus',
     'native_executed_producer_source':'flyvis_assets/producer_native_executed.py',
     'conditions':{}}
for stem in ['flyvis_trained_dimensionless','flyvis_trained_dt001','flyvis_trained_dt0005']:
    d=json.loads((root/f'{stem}.json').read_text())
    path=root/f'{stem}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==d['arrays_sha256']
    a=np.load(path)
    assert [(e['group'],e['index']) for e in d['coordinate_manifest']]==identities
    np.testing.assert_array_equal(a['interior_mask'],mask)
    for e,n in zip(d['coordinate_manifest'],native['coordinate_manifest']):
        assert e==n
    dt=d['measurement']['dt']
    repeat=int(round(.02/dt))
    np.testing.assert_array_equal(a['movie'],np.repeat(baseline['movie'],repeat,axis=1))
    J=a['J_raw_step_0.01']/baseline['output_scales'][:,None]
    grids={}
    offset=0
    for g in [64,32,16,8,6]:
        grids[str(g)]=spectrum(J[offset:offset+g,mask])
        offset+=g
    entry={'source_json':f'{stem}.json','common_weight_readout_sweep':grids,
           'own_weight_64_interior':d['spectra_by_step']['0.01']['64']['interior'],
           'baseline_relative_difference':float(np.linalg.norm(a['baseline']-baseline['baseline'])/np.linalg.norm(baseline['baseline'])),
           'jacobian_difference_from_native':comparison(J[:64,mask],reference[:64,mask]),
           'native_clipped_time_constant_count':sum(e['group']=='nodes_time_const' and e['baseline']<dt for e in d['coordinate_manifest'])}
    out['conditions'][f'{dt:g}']=entry
J1=np.load(root/'flyvis_trained_dt001.npz')['J_raw_step_0.01']/baseline['output_scales'][:,None]
J2=np.load(root/'flyvis_trained_dt0005.npz')['J_raw_step_0.01']/baseline['output_scales'][:,None]
out['comparison_005_vs_01']=comparison(J2[:64,mask],J1[:64,mask])
(root/'flyvis_timestep_control.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
for dt,r in out['conditions'].items():
    s=r['common_weight_readout_sweep']['64']
    print(dt,s['eff_dim_90'],s['eff_dim_99'],s['participation_ratio'],r['baseline_relative_difference'],r['jacobian_difference_from_native']['relative_frobenius'])
print('005 vs01 Frobenius',out['comparison_005_vs_01']['relative_frobenius'])
