"""Check tracked source-export files before publication (no upstream traversal)."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import pickletools
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forbidden-text', action='append', default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entries = subprocess.check_output(['git', 'ls-files', '--stage', '-z'], cwd=root).split(b'\0')
    errors = []
    checked = 0
    sensitive = re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}')
    for entry in filter(None, entries):
        meta, raw_path = entry.split(b'\t', 1)
        if meta.startswith(b'160000 '):
            continue
        name = raw_path.decode()
        path = root / name
        if path.suffix == '.pt' and name in {
            'checkpoints/samsone_clm_common/calibrated.pt',
            'checkpoints/samsone_clm_music/calibrated.pt',
        }:
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if member.endswith('/data.pkl'):
                        strings = [arg for op, arg, pos in pickletools.genops(archive.read(member)) if isinstance(arg, str)]
                        for value in strings:
                            if '/home/' in value or '/media/' in value or any(x.casefold() in value.casefold() for x in args.forbidden_text):
                                errors.append(name + ': identifying checkpoint metadata')
            continue
        if path.suffix in {'.pt', '.ckpt', '.safetensors', '.wav', '.mp3', '.flac', '.png'}:
            errors.append(name + ': binary/data file outside publication scope')
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            errors.append(name + ': unexpected non-text content')
            continue
        checked += 1
        if re.search(r'[\u0400-\u04ff]', text):
            errors.append(name + ': untranslated Cyrillic text')
        if sensitive.search(text):
            errors.append(name + ': possible credential')
        for value in args.forbidden_text:
            if value.casefold() in text.casefold():
                errors.append(name + ': forbidden identifying text')
        if path.suffix == '.json':
            try:
                json.loads(text)
            except ValueError:
                errors.append(name + ': invalid JSON')
    print(json.dumps({'checked_text_files': checked, 'errors': errors}, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
