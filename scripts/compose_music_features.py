"""Join SHA-verified frozen feature stores, with cached common/music replay.

No audio or encoder is needed for replay. Each component retains the exact
paired correct/random/silence features used in its own evaluation cohort.
"""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from samsone_clm.common import ROOT, ORIGINAL_DATA_ROOT, digest, sha256, write_json
from samsone_clm.precompute import load_features
from samsone_clm.staged_train import ENCODER_FIELDS
from samsone_clm.resources import require_resources


def select_component(manifest, fraction=1., balanced_heldout=False, music_only=False):
    rows = manifest['rows']
    groups = defaultdict(list)
    for i,row in enumerate(rows):
        if music_only and row['question_type'].lower() != 'music':
            continue
        if row.get('training_role') == 'contrast_only':
            continue
        if row['split'] != 'train' and not manifest.get('ablation_available',[True]*len(rows))[i]:
            continue
        answer = row['choices'][row['answer_index']]
        key = (row['split'],row.get('music_task',row['question_type']),answer if balanced_heldout and row['split'] != 'train' else '')
        groups[key].append(i)
    selected = []
    for (split,task,label), values in sorted(groups.items()):
        if split != 'train' and balanced_heldout and any(rows[i].get('genre_corpus') for i in values):
            parents = defaultdict(list)
            for i in values:
                parents[rows[i].get('parent_sample_id',rows[i]['audio_identity'])].append(i)
            ordered = sorted(parents,key=lambda parent:digest(['genre-evaluation-parent-v1',parent]))
            # 64 RECORDINGS per semantic class, all available question forms.
            for parent in ordered[:64]:
                selected.extend(parents[parent])
            continue
        count = max(1,round(len(values)*fraction)) if split == 'train' else min(len(values),64 if balanced_heldout else 128)
        selected.extend(sorted(values,key=lambda i:digest(['music-feature-cohort-v1',rows[i]['sample_id']]))[:count])
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=int,choices=[5,6],required=True)
    parser.add_argument('--new-manifests',nargs='+',required=True)
    parser.add_argument('--common-manifest',required=True)
    parser.add_argument('--stage5-manifest')
    parser.add_argument('--replay-train-ratio',type=float,default=1.,help='Stage6 music replay count relative to new train rows')
    parser.add_argument('--replay-window-manifest',help='Replace corresponding TRAIN replay rows; keep both held-out views')
    parser.add_argument('--pointer',type=Path,help='Separate pointer; preserve completed stage manifests')
    parser.add_argument('--shard-size',type=int,default=1024)
    args = parser.parse_args()
    require_resources()
    if args.stage == 6 and not args.stage5_manifest:
        parser.error('Stage 6 requires verified Stage 5 replay')
    components = []
    for path in args.new_manifests:
        data,manifest = load_features(path,dtype=torch.bfloat16)
        components.append((path,data,manifest,select_component(manifest,balanced_heldout=True)))
    new_train = sum(manifest['rows'][i]['split']=='train' for _,_,manifest,indices in components for i in indices)
    common_data,common = load_features(args.common_manifest,dtype=torch.bfloat16)
    # Stage 5 80% new /20% common. Stage 6 45% new /45% Stage 5 /10% common.
    desired_common = round(new_train*(.25 if args.stage==5 else 10/45))
    common_train = sum(r['split']=='train' and r['question_type'].lower()=='music' and r.get('training_role')!='contrast_only' for r in common['rows'])
    fraction = min(1.,desired_common/max(1,common_train))
    components.append((args.common_manifest,common_data,common,select_component(common,fraction,music_only=True)))
    if args.stage == 6:
        replay_data,replay = load_features(args.stage5_manifest,dtype=torch.bfloat16)
        music_indices = [i for i,r in enumerate(replay['rows']) if r.get('music_task') and r['split']=='train']
        music_indices.sort(key=lambda i:digest(['stage6-replay-v1',replay['rows'][i]['sample_id']]))
        # If the new corpus is larger, replay saturates at all independent music
        # QA; do not silently duplicate rows to pretend to increase data size.
        heldout = [i for i,r in enumerate(replay['rows']) if r.get('music_task') and r['split']!='train']
        selected=music_indices[:round(new_train*args.replay_train_ratio)]+heldout
        if args.replay_window_manifest:
            window_data,window=load_features(args.replay_window_manifest,dtype=torch.bfloat16)
            parents={r['sample_id']:r for r in replay['rows']}
            selected_parents={replay['rows'][i]['sample_id'] for i in selected}
            window_indices=[];replaced=set()
            for i,row in enumerate(window['rows']):
                parent_id=row['parent_sample_id']
                if parent_id not in parents:raise ValueError('Window QA parent is absent from Stage5')
                parent=parents[parent_id]
                if any(row.get(k)!=parent.get(k) for k in ['question','choices','answer_index','split','split_group','audio_identity','music_task']):
                    raise ValueError('Window QA changes a source split or label')
                if row.get('music_view')!='serving-window':raise ValueError('Expected an explicitly marked serving window')
                if parent_id not in selected_parents:continue
                if row['split']!='train' and not bool(window_data['ablation_available'][i]):
                    raise ValueError('A held-out window lacks matched audio controls')
                window_indices.append(i)
                if row['split']=='train':replaced.add(parent_id)
            selected=[i for i in selected if replay['rows'][i]['sample_id'] not in replaced]
            components.append((args.replay_window_manifest,window_data,window,window_indices))
        components.append((args.stage5_manifest,replay_data,replay,selected))
    provenance = components[0][2]['provenance']
    for _,_,manifest,_ in components:
        for field in ENCODER_FIELDS:
            if manifest['provenance'].get(field) != provenance.get(field):
                raise ValueError(f'Frozen encoder mismatch: {field}')
    rows, tensors, choice_vectors, seen, splits = [],defaultdict(list),{},set(),{}
    component_cards = []
    ablation = []
    for path,data,manifest,indices in components:
        added = []
        for i in indices:
            row = manifest['rows'][i]
            if row['sample_id'] in seen:
                continue
            seen.add(row['sample_id'])
            for identity in (row['audio_identity'],row.get('pcm_identity',row['audio_identity']),row.get('split_group',row['audio_identity'])):
                previous = splits.setdefault(identity,row['split'])
                if previous != row['split']:
                    raise ValueError(f'Cross-component split leak: {identity}')
            added.append(i)
            rows.append(row)
            ablation.append(bool(data['ablation_available'][i]))
        for mode in ('correct','random','silence'):
            tensors[mode].append(data[mode][added])
        for text,value in zip(manifest['choices'],data['actions']):
            if text in choice_vectors and not torch.equal(choice_vectors[text],value):
                # FP32 extraction can have tiny batch-dependent rounding. All
                # components must share the verified encoder; pick first bytes.
                difference=(choice_vectors[text].float()-value.float()).norm()/value.float().norm().clamp_min(1e-8)
                if difference > .001:
                    raise ValueError(f'Incompatible cached action: {text}')
            choice_vectors.setdefault(text,value)
        component_cards.append(dict(path=str(path),manifest_digest=digest(manifest),selected_rows=len(added),
                                    split_counts=dict(Counter(rowspec['split'] for rowspec in (manifest['rows'][i] for i in added)))))
    choices = sorted({text for row in rows for text in row['choices']})
    if any(row['question_type'].lower() != 'music' for row in rows):
        raise ValueError('Stages 5/6 must contain only musical QA, including replay')
    mapping = {text:i for i,text in enumerate(choices)}
    tensors = {mode:torch.cat(values) for mode,values in tensors.items()}
    tensors['actions'] = torch.stack([choice_vectors[text] for text in choices])
    new_provenance = {field:provenance[field] for field in ENCODER_FIELDS}
    new_provenance.update(dataset_semantic_sha256=digest(rows),choices_sha256=digest(choices),
            random_recipe='paired ablations retained from each SHA-verified component; component-local identity derangement',
            ablation_splits=['validation','test'],music_stage=args.stage,components=component_cards)
    directory = ROOT/'features'/('music-stage'+str(args.stage)+'-'+digest(new_provenance)[:16])
    if (directory/'manifest.json').exists():
        load_features(directory/'manifest.json',dtype=torch.bfloat16)
        write_json(args.pointer or ROOT/f'features/music_stage{args.stage}_manifest.json',
                   dict(path=str(ORIGINAL_DATA_ROOT/directory.relative_to(ROOT)/'manifest.json'),sha256=sha256(directory/'manifest.json')))
        print(directory/'manifest.json')
        return
    directory.mkdir(parents=True,exist_ok=True)
    for mode,values in tensors.items():
        for first in range(0,len(values),args.shard_size):
            target = directory/f'{mode}-{first:06d}.safetensors'
            temporary = target.with_suffix('.tmp')
            save_file({'hidden':values[first:first+args.shard_size].contiguous()},str(temporary))
            temporary.replace(target)
    logical_directory = ORIGINAL_DATA_ROOT/directory.relative_to(ROOT)
    manifest = dict(provenance=new_provenance,directory=str(logical_directory),rows=rows,choices=choices,
                    candidate_indices=[[mapping[text] for text in row['choices']] for row in rows],
                    shard_size=args.shard_size,ablation_available=ablation,
                    components=component_cards,encoder_calls=0,music_only=True,
                    shard_sha256={p.name:sha256(p) for p in directory.glob('*.safetensors')})
    write_json(directory/'manifest.json',manifest)
    write_json(args.pointer or ROOT/f'features/music_stage{args.stage}_manifest.json',dict(path=str(logical_directory/'manifest.json'),sha256=sha256(directory/'manifest.json')))
    write_json(directory/'composition_report.json',dict(rows=len(rows),split_counts=dict(Counter(r['split'] for r in rows)),
               music_task_counts=dict(Counter(r.get('music_task','common_replay') for r in rows)),
               components=component_cards,encoder_calls=0,split_leaks=0))
    load_features(directory/'manifest.json',dtype=torch.bfloat16)
    print(directory/'manifest.json',flush=True)


if __name__ == '__main__':
    main()
