"""Exercise the running demo with a real source WAV and padded playback windows."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import requests

from samsone_clm.common import ROOT, read_rows, resolve_data_path, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8800")
    parser.add_argument("--dataset", default=str(ROOT/"processed/stage1_10k.jsonl"))
    parser.add_argument('--feature-manifest',help='Use held-out rows from this manifest instead of dataset')
    parser.add_argument('--tasks',nargs='+',default=['instrument_family','vocals','tempo'])
    parser.add_argument('--checkpoint-sha256',help='Require this exact served checkpoint')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/web_demo_proof_latest.json')
    args = parser.parse_args()
    url = args.url.rstrip("/")
    session = requests.Session()
    if urlparse(url).hostname in {'localhost','127.0.0.1','::1'}:
        session.trust_env=False
    health = session.get(url+"/api/health", timeout=60)
    health.raise_for_status()
    assert health.json()["samsone_frozen"]
    if args.checkpoint_sha256:
        assert health.json()['checkpoint_sha256']==args.checkpoint_sha256
    home = session.get(url, timeout=30)
    home.raise_for_status()
    assert "<audio" in home.text and "Analyze" in home.text
    if args.feature_manifest:
        manifest=json.loads(resolve_data_path(args.feature_manifest).read_text())
        rows=[next(r for r in manifest['rows'] if r['split']=='test' and r.get('music_task')==task)
              for task in args.tasks]
    else:
        rows=[read_rows(resolve_data_path(args.dataset))[0]]
    results = []
    for row in rows:
        with resolve_data_path(row['audio_path']).open('rb') as source:
            upload=session.post(url+'/api/upload',files={'file':('real-source.wav',source,'audio/wav')},timeout=90)
        upload.raise_for_status()
        uploaded=upload.json()
        playback=session.get(url+uploaded['audio_url'],headers={'Range':'bytes=0-1023'},timeout=30)
        playback.raise_for_status()
        assert playback.status_code==206 and len(playback.content)==1024
        for position in (0.,uploaded['duration_s']/2,uploaded['duration_s']):
            payload=dict(audio_id=uploaded['audio_id'],position_s=position,question=row['question'],choices=row['choices'])
            response=session.post(url+'/api/analyze',json=payload,timeout=90)
            response.raise_for_status()
            result=response.json()
            assert result['window']['total_seconds']==10.
            assert abs(sum(item['probability'] for item in result['ranking'])-1)<1e-5
            assert {item['choice'] for item in result['ranking']}==set(row['choices'])
            if position in (0.,uploaded['duration_s']):
                assert result['window']['silence_before_s' if position==0. else 'silence_after_s']==5.
            if position==uploaded['duration_s']/2:
                reverse=session.post(url+'/api/analyze',json={**payload,'choices':list(reversed(row['choices']))},timeout=90)
                reverse.raise_for_status()
                scores={item['choice']:item['probability'] for item in result['ranking']}
                assert all(abs(item['probability']-scores[item['choice']])<1e-5 for item in reverse.json()['ranking'])
            results.append(dict(task=row.get('music_task'),source_audio=row['audio_path'],**result))
    report=dict(passed=True,health=health.json(),samples=len(results),seek_range_verified=True,
                choice_permutation_verified=True,results=results,
                scope='Functional ten-second window checks; full-clip labels are not scored on cropped windows')
    write_json(args.output,report)
    print(f"Web verification passed: {len(results)} real-audio windows, range seek, silence padding, choice permutation; {results[-1]['checkpoint']}")


if __name__ == "__main__":
    main()
