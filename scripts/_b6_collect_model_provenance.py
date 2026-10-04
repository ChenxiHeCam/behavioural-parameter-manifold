"""Hash the model sources and runtime used for a new BAAI rerun."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

root = Path(os.environ.get('BAAI_ROOT', '/root/autodl-tmp/BAAIWorm_fresh'))
out = Path(os.environ['PAPER2_OUTPUT_DIR'])
out.mkdir(parents=True, exist_ok=True)
selected = [root/'eworm/network/config.json', root/'build_headless/build/interact.so',
            root/'build_headless/build/libsim.so', root/'build_headless/build/libfem.so',
            root/'build_headless/interact_headless.cpp', root/'Metaworm/sim/Environment.cpp']
for folder in ('eworm/network', 'eworm/ghost_in_mesh_sim', 'eworm/components/param',
               'eworm/components/model', 'eworm/components/mechanism/modfile'):
    for p in (root/folder).rglob('*'):
        if p.is_file() and p.suffix in ('.py','.json','.hoc','.mod','.pkl','.npy'):
            selected.append(p)
compiled = Path(os.environ.get('BAAI_MECHANISM_DIR', str(root/'eworm/components/mechanism')))
selected += list(compiled.rglob('libnrnmech.so'))
for folder in ('Metaworm/data', 'Metaworm/args'):
    selected += [p for p in (root/folder).rglob('*') if p.is_file() and 'output' not in p.parts]
for folder in ('Metaworm/sim', 'build_headless'):
    selected += [p for p in (root/folder).rglob('*') if p.is_file() and
                 (p.suffix in ('.cpp','.h','.hpp') or p.name=='CMakeLists.txt')]
files = {}
for p in sorted(set(selected)):
    if p.is_file():
        files[str(p)] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
for p in Path(__file__).parent.glob('_*.py'):
    files[str(p)] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
metadata = {'model_files': files, 'python': sys.version,
            'historical_backend': 'unavailable historical GUI runtime; original ports refuse connections',
            'current_backend': 'headless native create_environment / physics_step(worm_id=0)',
            'current_backend_validation': '10 closed-loop steps after10initialization steps succeeded; physics_step_full crashes and is excluded',
            'scope': 'new headless rerun; agreement with historical scalar results must be assessed separately',
            'library_preload': os.environ.get('LD_PRELOAD'),
            'thread_environment': {k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}}
(out/'model_runtime_provenance.json').write_text(json.dumps(metadata, indent=2))
with zipfile.ZipFile(out/'B6_model_and_runtime_sources.zip','w',zipfile.ZIP_DEFLATED) as archive:
    archive_manifest = {}
    for path, info in files.items():
        p = Path(path)
        name = 'model/'+str(p.relative_to(root)).replace('\\','/') if p.is_relative_to(root) else 'runtime/'+str(p.relative_to(Path(__file__).parent)).replace('\\','/')
        archive.write(p,name)
        archive_manifest[name]=info
    archive.writestr('source_manifest.json',json.dumps(archive_manifest,indent=2))
(out/'B6_source_manifest.json').write_text(json.dumps(archive_manifest,indent=2))
print('model/source/runtime files hashed:', len(files))
