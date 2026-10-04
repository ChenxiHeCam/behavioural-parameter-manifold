"""Preserve an omitted runtime dependency without changing the locked experiment."""
from pathlib import Path
import hashlib, json, zipfile

root = Path(__file__).resolve().parent
out = root / 'formal_results_20261004'
sha = lambda b: hashlib.sha256(b).hexdigest()
runtime = (root / 'run_pilot.py').read_bytes()
with zipfile.ZipFile(root / 'pilot_evidence.zip') as z:
    archived = z.read('runtime/run_pilot.py')
assert runtime == archived, 'Runtime dependency changed since the preserved pilot'
snapshot_before = sha((out / 'source_snapshot.zip').read_bytes())
protocol_before = sha((out / 'protocol_locked.json').read_bytes())
dep = out / 'supplemental_runtime_dependencies'
dep.mkdir(exist_ok=True)
(dep / 'run_pilot.py').write_bytes(runtime)
provenance = {
    'reason': 'run_formal.py imports GROUPS and stimulus from run_pilot.py; the locked snapshot omitted this dependency.',
    'run_pilot_sha256': sha(runtime),
    'matches_preserved_pilot_runtime_byte_for_byte': True,
    'locked_source_snapshot_sha256_unchanged': snapshot_before,
    'locked_protocol_sha256_unchanged': protocol_before,
    'replay_instruction': 'Extract source_snapshot.zip to a runtime directory, then copy supplemental_runtime_dependencies/run_pilot.py into that directory. Use the separately retained model bundle and the recorded environment.',
    'scope': 'Post-run provenance supplement only; no protocol, result, or runtime source was edited.'
}
(dep / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
(dep / 'package_dependency_supplement.py').write_bytes(Path(__file__).read_bytes())
inventory = {str(p.relative_to(out)): {'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
             for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'inventory.json'}
(out / 'inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
target = root / 'formal_evidence_20261004.zip'
print('Packaging supplemented evidence', flush=True)
with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file(): z.write(p, p.relative_to(root))
    for name in ('verify_formal.py','summarize_formal.py','check_formal_fresh_parity.py','collect_formal.py'):
        z.write(root/name, Path('formal_results_20261004/verification_sources')/name)
    for name in ('persistent_backend_parity.json','nonlinear_development_report.json','development_ensemble.npz'):
        z.write(root/'formal_development'/name, Path('formal_development')/name)
assert sha((out/'source_snapshot.zip').read_bytes()) == snapshot_before
assert sha((out/'protocol_locked.json').read_bytes()) == protocol_before
manifest = {'sha256': sha(target.read_bytes()), 'bytes': target.stat().st_size,
            'all_case_outputs_included': True, 'formal_directory': out.name,
            'includes_development_sanity_checks': True, 'runtime_dependency_supplement': provenance}
(root/'formal_evidence_20261004_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest), flush=True)
