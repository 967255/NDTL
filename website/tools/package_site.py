#!/usr/bin/env python3
"""Bundle local research JSON for offline browsing and package the static website."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'dist'


def build(make_zip: bool = False) -> None:
    inputs = {'fusion': 'fusion.json', 'classification': 'classification.json',
              'applications': 'applications.json', 'features': 'feature-analysis.json',
              'landscape': 'classification-landscape.json'}
    data = {key: json.loads((DIST / 'data' / name).read_text()) for key, name in inputs.items()}
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    (DIST / 'data' / 'research-data.js').write_text('window.NDTL_DATA = ' + payload + ';\n', encoding='utf-8')
    manifest = {str(path.relative_to(DIST)): {'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(DIST.rglob('*')) if path.is_file() and path.name != 'site-manifest.json'}
    (DIST / 'site-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if make_zip:
        destination = ROOT / 'ndtl-website-preview.zip'
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(ROOT.rglob('*')):
                if path.is_file() and path.suffix != '.zip' and not any(part in {'__pycache__', '.git', 'node_modules'} for part in path.relative_to(ROOT).parts):
                    archive.write(path, 'website/' + str(path.relative_to(ROOT)))
        print(destination)
    print(f'Static website: {DIST}; {len(manifest)} files; research bundle {len(payload.encode()):,} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    archive = parser.add_mutually_exclusive_group()
    archive.add_argument('--zip', dest='make_zip', action='store_true', help='Also create a preview archive')
    archive.add_argument('--no-zip', dest='make_zip', action='store_false', help='Build static files only (default)')
    parser.set_defaults(make_zip=False)
    args = parser.parse_args()
    build(args.make_zip)
