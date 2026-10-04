"""Verify and package all formal cases, including optimization failures."""
import os
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile

root=Path(__file__).resolve().parent
directory=root/'formal_results_20261004'
if not (directory/'formal_report.json').exists():raise RuntimeError('Formal experiment has not finished')
subprocess.run([sys.executable,'-u',str(root/'check_formal_fresh_parity.py')],check=True)
subprocess.run([sys.executable,'-u',str(root/'verify_formal.py'),str(directory)],check=True)
subprocess.run([sys.executable,'-u',str(root/'summarize_formal.py'),str(directory)],check=True)
print('Creating file inventory',flush=True)
inventory={str(p.relative_to(directory)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='inventory.json'}
(directory/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
target=root/'formal_evidence_20261004.zip'
print('Packaging '+str(len(inventory))+' files',flush=True)
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(directory.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(root))
    for name in ('verify_formal.py','summarize_formal.py','check_formal_fresh_parity.py','collect_formal.py'):
        z.write(root/name,Path('formal_results_20261004/verification_sources')/name)
    for name in ('persistent_backend_parity.json','nonlinear_development_report.json','development_ensemble.npz'):
        z.write(root/'formal_development'/name,Path('formal_development')/name)
manifest={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'all_case_outputs_included':True,'formal_directory':'formal_results_20261004','includes_development_sanity_checks':True}
(root/'formal_evidence_20261004_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PACKAGED '+json.dumps(manifest),flush=True)
