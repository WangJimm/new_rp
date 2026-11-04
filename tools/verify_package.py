"""Verify the shipped snapshot with Python's standard library; no training."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    errors=[]
    for row in manifest['files']:
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()): raise SystemExit('Invalid manifest path')
        if not path.is_file(): errors.append(row['path']+': missing')
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']: errors.append(row['path']+': changed')
    if errors:
        print('\n'.join(errors)); raise SystemExit(1)
    print(f"PASS: {len(manifest['files'])} files match the packaged snapshot.")
if __name__=='__main__': main()
