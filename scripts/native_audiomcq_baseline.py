"""Official frozen Samsone356M MCQ diagnostic, with conservative parsing and raw responses."""
import json
import re
import time

import numpy as np
import soundfile as sf
import torch

from samsone_clm.common import ROOT, digest, read_rows, write_json
from samsone_clm.encoder import FrozenSamsone
from samsone_clm.resources import require_resources, probe_storage
from samsone import Samsone356M


def parse_answer(response, choices):
    value = response.strip()
    match = re.match(r'^(?:answer\s*[:=]\s*)?\(?([A-Z])\)?(?:[.)\s:]|$)', value, re.I)
    if match and ord(match.group(1).upper())-65 < len(choices):
        return ord(match.group(1).upper())-65
    normalized = value.lower().strip(' .\n')
    exact = [i for i,c in enumerate(choices) if c.lower().strip(' .\n') == normalized]
    return exact[0] if len(exact) == 1 else None


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    directory = ROOT/'runs/v3/native_baseline'
    probe_storage(directory)
    rows = [r for r in read_rows(ROOT/'processed/v2/stage1_10k.jsonl') if r['split']=='validation']
    selected = []
    for kind in ('music','sound','speech','temporal'):
        selected.extend(sorted([r for r in rows if r['question_type']==kind], key=lambda r:digest([174,r['sample_id']]))[:25])
    # The original helper is the same API used in the README, explicitly in FP32.
    model = Samsone356M(dtype=torch.float32)
    model.model.eval().requires_grad_(False)
    responses = []
    silence_path = ROOT/'tmp/native_baseline_silence.wav'
    for i,row in enumerate(selected):
        require_resources()
        question = row['question'] + '\n' + '\n'.join(f'{chr(65+j)}. {c}' for j,c in enumerate(row['choices']))
        question += '\nChoose one answer. Reply only with its letter.'
        waveform,rate = sf.read(row['audio_path'], dtype='float32')
        sf.write(silence_path, np.zeros_like(waveform), rate, subtype='FLOAT')
        partner = next(r for r in selected[i+1:]+selected[:i] if r['audio_identity'] != row['audio_identity'])
        result = dict(sample_id=row['sample_id'], question_type=row['question_type'],
                      audio_path=row['audio_path'], question=row['question'], choices=row['choices'], answer_index=row['answer_index'], modes={})
        for mode,path in [('correct',row['audio_path']),('random',partner['audio_path']),('silence',str(silence_path))]:
            start = time.monotonic()
            response = model(audio=path, prompt=question, max_new_tokens=32)
            predicted = parse_answer(response,row['choices'])
            result['modes'][mode] = dict(response=response,predicted_index=predicted,
                                         correct=predicted==row['answer_index'], latency_s=time.monotonic()-start)
        responses.append(result)
        report = dict(status='running', model='Samsone356M', dtype='float32', protocol='explicit choices; letter or exact-choice parsing; invalid responses count as incorrect',
                      samples=len(responses), cases=responses, frozen=all(not p.requires_grad for p in model.model.parameters()))
        write_json(directory/'results.json',report)
        print(json.dumps(dict(done=i+1, modes=result['modes']),ensure_ascii=False),flush=True)
    del model
    torch.cuda.empty_cache()
    encoder = FrozenSamsone()
    proof = []
    for row in selected[::25]:
        captured = {}
        def capture(module,inputs,kwargs,output):
            if not captured:
                captured['hidden'] = encoder._last_valid(output.last_hidden_state,kwargs['attention_mask']).detach().cpu()
        handle=encoder.transformer.register_forward_hook(capture,with_kwargs=True)
        try:
            encoder.helper(audio=row['audio_path'],prompt=row['question'],max_new_tokens=1)
        finally:
            handle.remove()
        hidden = encoder.state([row['audio_path']],[row['question']])
        error=float((hidden-captured['hidden']).abs().max())
        assert error < 1e-4, error
        proof.append(dict(sample_id=row['sample_id'], native_hidden_max_error=error))
    report['status']='completed'
    report['extraction_proof']=proof
    report['summary']={mode:{kind:dict(samples=len(group), accuracy=sum(r['modes'][mode]['correct'] for r in group)/len(group),
            parse_coverage=sum(r['modes'][mode]['predicted_index'] is not None for r in group)/len(group))
            for kind in ('overall','music','sound','speech','temporal')
            for group in [[r for r in responses if kind=='overall' or r['question_type']==kind]]}
            for mode in ('correct','random','silence')}
    write_json(directory/'results.json',report)
    print(json.dumps(report['summary']),flush=True)


if __name__=='__main__':
    main()
