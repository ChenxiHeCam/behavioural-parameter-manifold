"""Record matched parameter subsets and coordinate sensitivity from retained J."""
import json
from pathlib import Path
import numpy as np
from flyvis_sensitivity_revision import spectrum, comparison

root=Path(__file__).resolve().parents[1]/'results/revision_20261004'
states={s:json.loads((root/f'flyvis_{s}_dimensionless.json').read_text()) for s in ['trained','initialized']}
arrays={s:np.load(root/f'flyvis_{s}_dimensionless.npz') for s in states}
identities={s:[(e['group'],e['index']) for e in d['coordinate_manifest']] for s,d in states.items()}
assert identities['trained']==identities['initialized'], 'Parameter identities/order differ between model states'
mask=arrays['trained']['interior_mask']
out={'comparison_scope':'one trained network and its default initialization; no ensemble or causal claim',
     'historical_result':'E4_flyvis_readout.json remains an initialized/raw-coordinate/330-parameter analysis',
     'matched_subset':'703 coordinates interior in the trained network, indexed identically in both states',
     'coordinate_comparison':'raw derivative estimated by local chain rule J_theta=J_z/scale, not a rerun of the old h=0.2 protocol',
     'common_output_scale':'trained network abs(unperturbed bin mean)+1e-6 for both states',
     'states':{}}
for state,d in states.items():
    J=arrays[state]['J_scaled_step_0.01'][:64]
    scales=np.array([e['scale'] for e in d['coordinate_manifest']])
    out['states'][state]={'matched_703_dimensionless':spectrum(J[:,mask]),
                          'all_734_dimensionless':spectrum(J),
                          'all_734_raw_coordinates_chain_rule':spectrum(J/scales[None,:])}
    common=arrays[state]['J_raw_step_0.01']/arrays['trained']['output_scales'][:,None]
    grids={}
    offset=0
    for g in [64,32,16,8,6]:
        grids[str(g)]=spectrum(common[offset:offset+g,mask])
        offset+=g
    out['states'][state]['common_weight_matched_703_readout_sweep']=grids
    out['states'][state]['common_weight_step_validation']={
        h:comparison((arrays[state][f'J_raw_step_{h}']/arrays['trained']['output_scales'][:,None])[:64,mask],common[:64,mask])
        for h in d['spectra_by_step']}
(root/'flyvis_state_coordinate_comparison.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
for state,results in out['states'].items():
    print(state,{k:(v['eff_dim_90'],v['eff_dim_99']) for k,v in results.items() if 'eff_dim_90' in v})
