"""Derive connection-category summaries and archive the actual rerun sources."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import zipfile
import numpy as np

root = Path(os.environ.get('BAAI_ROOT', '/root/autodl-tmp/BAAIWorm_fresh'))
out = Path(os.environ['PAPER2_OUTPUT_DIR'])
sys.path[:0] = [str(root), str(root/'eworm/ghost_in_mesh_sim')]
tuned = root/'eworm/ghost_in_mesh_sim/data/tuned/video_offline'
with open(tuned/'video_offline_abscircuit.pkl', 'rb') as f:
    circuit = pickle.load(f)
categories = np.array([c.category for c in circuit.connections])
result = json.loads((out/'E42_full_spectrum_rerun.json').read_text())
for readout, record in result['readouts'].items():
    with np.load(out/record['matrix_file']) as z:
        J = z['J']
    elasticity = np.linalg.norm(J, axis=0)
    record['elasticity_percentiles'] = {str(p): float(np.percentile(elasticity, p)) for p in (1,25,50,75,99)}
    record['n_parameters_with_no_measurable_effect'] = int((elasticity < 1e-12*max(elasticity.max(),1e-30)).sum())
    record['by_connection_category'] = {}
    for category in np.unique(categories):
        mask = categories == category
        record['by_connection_category'][category] = {'n': int(mask.sum()),
            'median_elasticity': float(np.median(elasticity[mask])),
            'share_of_curvature': float((elasticity[mask]**2).sum()/max((elasticity**2).sum(),1e-30))}
(out/'E42_full_spectrum_rerun.json').write_text(json.dumps(result,indent=2))

files = [root/'eworm/network/config.json', tuned/'video_offline_abscircuit.pkl',
         tuned/'video_offline_wout.pkl']
for folder, suffixes in [('eworm/network',('.py',)), ('eworm/utils',('.py',)),
                         ('eworm/ghost_in_mesh_sim',('.py',)),
                         ('eworm/components/param',('.json',)),
                         ('eworm/components/model',('.hoc',)),
                         ('eworm/components/mechanism/modfile',('.mod',))]:
    files.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in suffixes)
if (tuned/'video_offline_loc.npy').exists():
    files.append(tuned/'video_offline_loc.npy')
runtime = Path(__file__).parent
files.extend(p for p in runtime.glob('_*.py') if p.is_file())
files.extend(p for p in (runtime/'dispatch_smoke').glob('*.py') if p.is_file())
with zipfile.ZipFile(out/'E42_model_and_runtime_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
    metadata = {}
    for p in sorted(set(files)):
        name = 'model/'+str(p.relative_to(root)).replace('\\','/') if p.is_relative_to(root) else 'runtime/'+str(p.relative_to(runtime)).replace('\\','/')
        archive.write(p,name)
        metadata[name] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
    archive.writestr('source_manifest.json',json.dumps(metadata,indent=2))
(out/'E42_source_manifest.json').write_text(json.dumps(metadata,indent=2))
print('finalized full spectra and archived',len(files),'source files',flush=True)
