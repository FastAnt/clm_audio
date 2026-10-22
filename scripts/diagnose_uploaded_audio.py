"""Inspect ranking/audio controls on one exact demo window; no training or selection."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from samsone_clm.audio import decode_audio
from samsone_clm.common import ROOT, resolve_data_path, sha256, write_json
from samsone_clm.infer import Ranker
from samsone_clm.resources import require_resources
from samsone_clm.server import audio_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--position', type=float, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--compare-checkpoint', type=Path)
    parser.add_argument('--question', default='what is music genre?')
    parser.add_argument('--choices', nargs='+', default=['rock', 'hip hop', 'classic instrumental'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_resources()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = resolve_data_path(args.audio)
    waveform = decode_audio(source)
    window, details = audio_window(waveform, args.position)
    temporary = ROOT / 'tmp' / ('diagnose-' + sha256(source)[:12])
    temporary.mkdir(parents=True, exist_ok=True)
    paths = {'correct': temporary / 'correct.wav', 'silence': temporary / 'silence.wav'}
    sf.write(paths['correct'], window, 16000, subtype='FLOAT')
    sf.write(paths['silence'], np.zeros_like(window), 16000, subtype='FLOAT')
    # The exact taxonomy/question trained from Jamendo human consensus.
    canonical_choices = ['alternative', 'blues', 'electronic', 'folk country',
                         'funk soul rnb', 'jazz', 'pop', 'rap hiphop', 'rock']
    cases = [
        ('user', args.question, args.choices),
        ('same_choices_canonical_question', 'Which broad musical genre best describes this excerpt?', args.choices),
        ('expanded_choices_user_question', args.question, canonical_choices),
        ('canonical', 'Which broad musical genre best describes this excerpt?', canonical_choices),
    ]
    ranker = Ranker(resolve_data_path(args.checkpoint))
    report = dict(scope='Single uploaded file diagnostic; no known genre label; no training/checkpoint selection',
                  audio_sha256=sha256(source), duration_s=len(waveform)/16000, window=details,
                  source_path=str(source), rankings=[], native_captions=[])
    for checkpoint in [args.checkpoint] + ([args.compare_checkpoint] if args.compare_checkpoint else []):
        ranker.reload(resolve_data_path(checkpoint))
        for name, question, choices in cases:
            result = dict(checkpoint_sha256=sha256(checkpoint), checkpoint=str(checkpoint),
                          case=name, question=question, choices=choices, modes={})
            for mode, path in paths.items():
                result['modes'][mode] = ranker.rank(path, question, choices)
            report['rankings'].append(result)
            write_json(args.output, report)
    # Official Samsone356M helper, using the same frozen model already loaded.
    for prompt in ['Caption the audio', 'Describe the music heard.',
                   'Does this music contain singing or voice, or is it instrumental?']:
        for mode, path in paths.items():
            started = time.monotonic()
            with torch.no_grad():
                response = ranker.encoder.helper(audio=str(path), prompt=prompt,
                                                 max_new_tokens=96)
            report['native_captions'].append(dict(mode=mode, prompt=prompt, response=response,
                                                 seconds=time.monotonic()-started))
            write_json(args.output, report)
    write_json(args.output, report)
    for result in report['rankings']:
        print(json.dumps(dict(checkpoint=result['checkpoint'], case=result['case'],
                              top={mode: values[:3] for mode, values in result['modes'].items()})), flush=True)
    print('Diagnostic saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
