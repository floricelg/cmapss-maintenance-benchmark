"""Download or import the pinned NASA archive and verify the twelve numeric files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_files(folder, expected):
    missing = []
    for name, sha in expected.items():
        if Path(name).name != name or not name.endswith('.txt'):
            raise ValueError('Unexpected filename in data manifest')
        target = folder / name
        if not target.exists():
            missing.append(name)
        elif digest(target) != sha:
            raise ValueError(f'{target.name} has a different SHA256. No file was overwritten.')
    return missing


def import_archive(archive, folder, manifest):
    if digest(archive) != manifest['archive_sha256']:
        raise ValueError('The ZIP checksum differs from the pinned NASA archive.')
    expected = manifest['files']
    missing = verify_files(folder, expected)
    verified = {}
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            name = PurePosixPath(entry.filename).name
            if name not in expected:
                continue
            if entry.is_dir() or '..' in PurePosixPath(entry.filename).parts or entry.file_size > MAX_ARCHIVE_BYTES:
                raise ValueError('Unsafe or oversized archive member')
            if name in verified:
                raise ValueError(f'Duplicate archive member {name}')
            content = zipped.read(entry)
            if hashlib.sha256(content).hexdigest() != expected[name]:
                raise ValueError(f'Unexpected data checksum for {name}')
            verified[name] = content
    if set(verified) != set(expected):
        raise ValueError('The archive does not contain all twelve required numeric files.')
    folder.mkdir(parents=True, exist_ok=True)
    for name in missing:
        with (folder / name).open('xb') as handle:
            handle.write(verified[name])
    if verify_files(folder, expected):
        raise ValueError('Some input files are missing after import')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded CMAPSSData.zip')
    parser.add_argument('--verify-only', action='store_true', help='Check existing numeric files without network access')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data_manifest.json').read_text(encoding='utf-8'))
    folder = ROOT / 'date'
    missing = verify_files(folder, manifest['files'])
    if not missing:
        print('All twelve NASA numeric files match the pinned SHA256 checksums.')
        return
    if args.verify_only:
        raise SystemExit('Missing input files: ' + ', '.join(missing))
    if args.archive:
        import_archive(args.archive, folder, manifest)
    else:
        try:
            with tempfile.TemporaryDirectory(prefix='cmapss-download-') as temporary:
                archive = Path(temporary) / manifest['archive_name']
                request = urllib.request.Request(manifest['download_url'], headers={'User-Agent': 'cmapss-maintenance-benchmark/1.0'})
                with urllib.request.urlopen(request, timeout=60) as response, archive.open('xb') as output:
                    received = 0
                    while block := response.read(1024 * 1024):
                        received += len(block)
                        if received > MAX_ARCHIVE_BYTES:
                            raise ValueError('Download exceeds the expected maximum archive size')
                        output.write(block)
                import_archive(archive, folder, manifest)
        except (urllib.error.URLError, TimeoutError) as error:
            raise SystemExit(
                f'Download failed: {error}\nDownload CMAPSSData.zip from {manifest["record_url"]}\n'
                'Then run: python scripturi/download_data.py --archive "path/to/CMAPSSData.zip"'
            ) from error
    print('Imported and verified all twelve NASA numeric files.')


if __name__ == '__main__':
    main()
