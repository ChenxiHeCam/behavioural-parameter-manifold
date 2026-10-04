"""Retain source-based metadata and historical animal checks without alteration."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import cell_panel

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'revision_20261004'
OUT.mkdir(exist_ok=True)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source=ROOT/'results'/'mapk_showcase.json'
    show=json.loads(source.read_text())
    star=np.array(cell_panel.PANEL['MAPK'][2])
    actual=cell_panel.run_model('MAPK',star)
    recorded=np.asarray(show['traces']['truth'])
    error=float(np.max(np.abs(actual-recorded)))
    if error > 1e-8: raise ValueError(f'MAPK generating trace does not reproduce: {error}')
    t=np.linspace(0,cell_panel.PANEL['MAPK'][5],len(recorded))
    payload={'time_seconds':t.tolist(),'source_sha256':sha(source),
             'producer_sha256':sha(Path(cell_panel.__file__)),
             'generating_trace_max_abs_error':error,
             'example_meets_fit_threshold':bool(show['best_fit']['converged']),
             'example_output_distance':show['best_fit']['L_final']}
    (OUT/'mapk_trace_metadata.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    provenance=[]
    for rel in ['two_connectome_consistency.json','card2_leifer/leifer_class_test.json']:
        old=Path(r'D:\Warm\animal_evidence')/rel
        dest=OUT/old.name
        shutil.copyfile(old,dest)
        provenance.append({'source':str(old),'destination':str(dest.relative_to(ROOT)),
                           'sha256':sha(old),'status':'unchanged historical exploratory comparison'})
    (OUT/'biological_checks_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(f'MAPK time axis verified against its model; {len(provenance)} historical checks retained.')

if __name__=='__main__': main()
