"""Publish a verified music export while preserving the previous model directory."""
import argparse
import json
from pathlib import Path
import shutil
import uuid

from samsone_clm.common import ROOT,sha256,write_json


def verify(directory):
    checks=json.loads((directory/'checksums.json').read_text())
    for name,digest in checks.items():
        if Path(name).name!=name or sha256(directory/name)!=digest:
            raise ValueError(f'Invalid music export file: {name}')
    card=json.loads((directory/'card.json').read_text())
    if not all(card.get(k) for k in ['music_only','encoder_frozen','both_original_heads_train','shared_adapter']):
        raise ValueError('Music export does not follow the required architecture')
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir',type=Path,required=True)
    parser.add_argument('--archive-name',required=True)
    args=parser.parse_args()
    if Path(args.archive_name).name!=args.archive_name or args.archive_name in {'.','..','samsone_clm_common','samsone_clm_music'}:
        raise ValueError('Archive must be a distinct model directory name')
    source=args.export_dir.absolute();checks=verify(source)
    target=ROOT/'models/samsone_clm_music';archive=ROOT/'models'/args.archive_name
    if target.exists() and verify(target)['best.pt']==checks['best.pt']:
        print('Selected music model already published');return
    if archive.exists():raise ValueError('Previous model archive already exists')
    temporary=target.with_name('.samsone_clm_music.'+uuid.uuid4().hex)
    shutil.copytree(source,temporary);verify(temporary)
    previous=None
    if target.exists():
        previous=verify(target)['best.pt'];target.rename(archive)
    try:temporary.rename(target)
    except BaseException:
        if previous:archive.rename(target)
        raise
    verify(target)
    write_json(ROOT/'runs/music/published_model.json',dict(source=str(source),target=str(target),
        best_sha256=checks['best.pt'],calibrated_sha256=checks['calibrated.pt'],
        previous_archive=str(archive) if previous else None,previous_best_sha256=previous,
        common_preserved=True))
    print(json.dumps({'published':str(target),'best_sha256':checks['best.pt']}))


if __name__=='__main__':main()
