"""Download public result assets without authentication and verify SHA256."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',action='store_true',help='Also download original response archives (~1.2 GB additional)')
    p.add_argument('--extract-prospective',action='store_true',help='Extract the downloaded prospective evidence archive')
    a=p.parse_args();manifest=json.loads(Path(__file__).with_name('release_assets.json').read_text())
    for row in manifest['assets']:
        israw=row['path'].endswith(('.tar','formal_evidence_20261004.zip'))
        if israw and not a.raw:continue
        target=(ROOT/row['path']).resolve()
        if not target.is_relative_to(ROOT):raise ValueError('Asset target leaves repository')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size==row['bytes'] and digest(target)==row['sha256']:
            print('Verified existing',row['path']);continue
        temp=target.with_name(target.name+'.part')
        print('Downloading',row['asset'],flush=True)
        request=urllib.request.Request(row['url'],headers={'User-Agent':'paper2-code-reproduction'})
        with urllib.request.urlopen(request,timeout=120) as response,temp.open('wb') as out:
            while True:
                block=response.read(1024*1024)
                if not block:break
                out.write(block)
        if temp.stat().st_size!=row['bytes'] or digest(temp)!=row['sha256']:
            raise ValueError('Download integrity check failed: '+row['asset'])
        temp.replace(target);print('Verified',row['path'])
    if a.extract_prospective:
        archive=ROOT/'results/prospective_20261004/formal_evidence_20261004.zip'
        if not archive.exists():raise FileNotFoundError('Use --raw to download the prospective archive first')
        directory=archive.parent
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                dest=(directory/name).resolve()
                if not dest.is_relative_to(directory):raise ValueError('Unsafe archive path')
            z.extractall(directory)
        print('Prospective evidence extracted.')
if __name__=='__main__':main()
