"""Restore historical Stage1–3 dashboard metadata from checked-in reports."""
import json
import shutil
from samsone_clm.common import PROJECT,ROOT,sha256,write_json


def main():
    for stage,name in [(1,'stage1'),(2,'stage2-audio-grounded'),(3,'stage3')]:
        source=PROJECT/'docs/results/v3'/name
        target=ROOT/'runs/history'/('stage'+str(stage))
        target.mkdir(parents=True,exist_ok=True)
        checks={}
        for name in ('metrics.json','progress.json','best.pt.info.json','training_curves.png'):
            original=source/name
            if not original.exists():continue
            expected=sha256(original)
            destination=target/name
            if not destination.exists():shutil.copyfile(original,destination)
            if sha256(destination)!=expected:raise ValueError('Conflicting historical dashboard file: '+str(destination))
            checks[name]=expected
        write_json(target/'checksums.json',checks)
        write_json(target/'history_source.json',dict(source=str(source),display_only=True,stage=stage,sha256=checks))
        print(json.dumps(dict(stage=stage,restored=str(target),metadata_files=len(checks))),flush=True)


if __name__=='__main__':main()
