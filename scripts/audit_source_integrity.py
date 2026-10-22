"""Audit unique source WAVs against frozen metadata before further training."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter,defaultdict

from samsone_clm.common import ROOT,read_rows,sha256,write_json
from samsone_clm.resources import require_resources


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',default=str(ROOT/'processed/v2/canonical.jsonl'))
    parser.add_argument('--output',type=Path,default=ROOT/'runs/v3/source_integrity.json')
    args=parser.parse_args()
    require_resources()
    rows=read_rows(args.dataset)
    groups=defaultdict(list)
    for row in rows:groups[row['audio_path']].append(row)
    def check(item):
        path,labels=item
        try:actual=sha256(Path(path));error=None
        except OSError as exception:actual=None;error=str(exception)
        expected={r['audio_sha256'] for r in labels}
        if expected=={actual}:return None
        return dict(path=path,expected_hashes=sorted(expected),actual_hash=actual,error=error,
                    source_dataset=labels[0]['source_dataset'],source_audio_id=labels[0]['source_audio_id'],
                    rows=labels)
    mismatches=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i,result in enumerate(pool.map(check,groups.items()),1):
            if result:mismatches.append(result)
            if i%1000==0:
                require_resources()
                print(json.dumps(dict(checked=i,total=len(groups),mismatches=len(mismatches))),flush=True)
    report=dict(dataset=args.dataset,unique_paths=len(groups),mismatched_paths=len(mismatches),
                mismatched_rows=sum(len(x['rows']) for x in mismatches),
                types=dict(Counter(r['source_dataset'] for x in mismatches for r in x['rows'])),mismatches=mismatches)
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='mismatches'}),flush=True)


if __name__=='__main__':main()
