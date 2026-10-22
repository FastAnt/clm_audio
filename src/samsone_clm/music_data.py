"""Music QA from explicit annotations. Captions never enter the state prompt."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile

import numpy as np
import soundfile as sf

from .common import ROOT, ORIGINAL_DATA_ROOT, digest, identity_split, sha256, write_json
from .resources import require_resources


def musicbench_parent(location):
    stem = PurePosixPath(location).stem
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}(?:_\d+)?', stem):
        raise ValueError(f'Invalid MusicBench source identity: {location}')
    return stem[:11]


def normalize_pitch(value):
    value = str(value).strip().replace('♯','#').replace('♭','b')
    pitches = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'Fb':4,
               'E#':5,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,
               'A#':10,'Bb':10,'B':11,'Cb':11,'B#':0}
    names = ['C','C sharp','D','E flat','E','F','F sharp','G','A flat','A','B flat','B']
    return names[pitches[value]] if value in pitches else None


def structure_labels(metadata, duration):
    """Filter automatic labels using confidence, timing, and beat consistency."""
    result = []
    beats = metadata.get('beats') or [[],[]]
    times, numbers = beats if len(beats) == 2 else ([], [])
    valid_beats = (len(times) == len(numbers) and len(times) >= 8 and
                   all(0 <= t <= duration+.08 for t in times) and
                   all(a < b for a,b in zip(times,times[1:])))
    if valid_beats:
        intervals = np.diff(times)
        bpm = metadata.get('bpm')
        beat_bpm = 60/float(np.median(intervals))
        if (isinstance(bpm,(int,float)) and np.isfinite(bpm) and 40 <= bpm < 240 and
            abs(beat_bpm-bpm)/bpm <= .05 and np.std(intervals)/np.mean(intervals) <= .25 and
            min(abs(bpm-edge) for edge in (80,120,160)) >= 4):
            choices = ['40 to 79 BPM','80 to 119 BPM','120 to 159 BPM','160 to 239 BPM']
            result.append(('tempo','What is the approximate tempo of the musical pulse in beats per minute?',
                           choices, choices[min(3,int((bpm-40)//40))] if bpm < 160 else choices[3],
                           dict(bpm=bpm,beat_bpm=beat_bpm)))
        integer = [int(n) for n in numbers]
        meter = max(integer)
        if (meter in (3,4) and len(integer) >= meter*3 and all(n == i for n,i in zip(numbers,integer)) and
            all(1 <= n <= meter for n in integer) and
            all(b == a%meter+1 for a,b in zip(integer,integer[1:]))):
            result.append(('meter','How many beats are there per bar in this music?',
                           ['three beats per bar','four beats per bar'],
                           'three beats per bar' if meter == 3 else 'four beats per bar',dict(beats_per_bar=meter)))
    key = metadata.get('key') or []
    confidence = metadata.get('keyprob') or []
    if len(key) == 2 and key[1] in ('major','minor') and confidence and min(confidence) >= .8:
        pitch = normalize_pitch(key[0])
        if pitch:
            pitches = [normalize_pitch(p) for p in ('C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B')]
            result.append(('key_tonic','What is the tonic note of the musical key?',pitches,pitch,
                           dict(confidence=min(confidence),original_key=key)))
            result.append(('key_mode','Is the overall musical key major or minor?',['major','minor'],key[1],
                           dict(confidence=min(confidence),original_key=key)))
    # No chord/progression labels inferred from captions or unconfident chords.
    # MusicBench chord annotations do not expose confidence. Preserve them as
    # metadata for auditing rather than silently declaring them reliable gold.
    return result


def waveform_record(path, identity, group, split, source, scope, window=None):
    audio, rate = sf.read(path, dtype='float32',always_2d=True)
    audio = audio.mean(1)
    if rate != 16000 or not len(audio) or len(audio) > 480000 or not np.isfinite(audio).all():
        raise ValueError(f'Invalid prepared audio: {path}')
    if np.sqrt(np.mean(audio.astype(np.float64)**2)) < 1e-6:
        raise ValueError(f'Effectively silent source: {path}')
    pcm = hashlib.sha256(audio.tobytes()).hexdigest()
    return dict(audio_path=str(ORIGINAL_DATA_ROOT/path.relative_to(ROOT)),audio_sha256=sha256(path),pcm_identity='pcm:'+digest([rate,pcm]),
                audio_identity=identity,source_recording_id=identity,split_group=group,split=split,
                duration_s=len(audio)/rate,audio_num_samples=len(audio),source_dataset=source,question_type='music',
                label_scope=scope,analyzed_window=window)


def qa(base, task, question, choices, answer, evidence):
    if answer not in choices or len(set(choices)) != len(choices) or len(choices) < 2:
        raise ValueError(f'Invalid options/label for {task}: {answer}')
    seed = digest([base['audio_identity'],task,question])
    ordered = sorted(choices,key=lambda text:digest([seed,text]))
    return dict(base,question=question,choices=ordered,answer_index=ordered.index(answer),music_task=task,
                label_evidence=evidence,sample_id=digest([base['audio_identity'],task,question,answer]))


def convert_audio(source,target,start=0.,duration=None):
    if target.exists():
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix('.wav.tmp')
    if Path(source).suffix.lower() == '.wav' and start == 0 and duration is None:
        info = sf.info(source)
        if info.samplerate == 16000 and info.channels == 1:
            # Native MusicBench/NSynth WAV requires no decode/resample round trip.
            shutil.copyfile(source,temporary)
            temporary.replace(target)
            return
    command = ['ffmpeg','-nostdin','-v','error','-threads','1','-ss',str(start),'-i',str(source)]
    if duration is not None:
        command += ['-t',str(duration)]
    command += ['-ac','1','-ar','16000','-c:a','pcm_f32le','-f','wav','-y',str(temporary)]
    subprocess.run(command,check=True,timeout=60,stdout=subprocess.DEVNULL)
    temporary.replace(target)


JAMENDO = {
    'genre_dortmund': ('genre','Which broad musical genre best describes this excerpt?',
                      ['alternative','blues','electronic','folk_country','funk_soul_rnb','jazz','pop','rap_hiphop','rock']),
    'genre_rosamerica': ('genre_rosamerica','Which musical style best describes this excerpt?',
                        ['classical','dance','hip_hop','jazz','pop','rhythm_and_blues','rock','speech']),
    'genre_tzanetakis': ('genre_gtzan','Which genre best fits the musical style of this excerpt?',
                        ['blues','classical','country','disco','hip_hop','jazz','metal','pop','reggae','rock']),
    'genre_electronic': ('electronic_genre','Which electronic music style best describes this excerpt?',
                         ['ambient','drum_and_bass','house','techno','trance']),
    'voice_instrumental': ('vocals','Does this music contain singing or voice, or is it instrumental?',
                           ['voice','instrumental']),
    'danceability': ('danceability','Is the musical pulse suitable for dancing?', ['danceable','not_danceable']),
}
for mood in ('acoustic','aggressive','electronic','happy','party','relaxed','sad'):
    JAMENDO['mood_'+mood] = ('mood_'+mood,f'Does this music have a {mood} character?', [mood,'not_'+mood])


def prepare_jamendo():
    root = ROOT/'raw/music/jamendo'
    report = json.loads((root/'provenance.json').read_text())
    if not report['complete']:
        raise ValueError('Jamendo downloads must finish before preparation')
    checksums = {line.split()[1].lstrip('*'):line.split()[0] for line in (root/'audio_sha256.txt').read_text().splitlines()}
    def prepare_track(line):
        result = []
        values = line.split('\t')
        track,artist,album,name,duration = values[:5]
        source_name = name.removesuffix('.mp3')+'.low.mp3'
        source = root/'audio'/source_name
        if sha256(source) != checksums[source_name]:
            raise ValueError(f'Published Jamendo source checksum mismatch: {source}')
        start = max(0.,(float(duration)-30)/2)
        target = ROOT/'processed/music/audio/jamendo'/f'{track}.wav'
        convert_audio(source,target,start=start,duration=30)
        group = 'jamendo-artist:'+artist
        try:
            base = waveform_record(target,'jamendo:'+track,group,identity_split(group),'MTG-Jamendo human consensus',
                      'full-track human label projected onto a central 30s excerpt; excerpt truth not separately annotated',
                      [start,start+30])
        except ValueError as error:
            return [],dict(track=track,reason=str(error),window=[start,start+30])
        for annotation in values[5:]:
            taxonomy,labels = annotation.split('---')
            labels = labels.split(',')
            if taxonomy not in JAMENDO or len(labels) != 3 or len(set(labels)) != 1:
                continue
            task,question,options = JAMENDO[taxonomy]
            if labels[0] not in options:
                continue
            labels_text = {label:label.replace('not_','not ').replace('_',' ') for label in options}
            result.append(qa(base,task,question,list(labels_text.values()),labels_text[labels[0]],
                           dict(taxonomy=taxonomy,annotators=3,agreement='unanimous',full_track_label=True)))
        return result,None
    rows,rejected = [],[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index,(prepared,rejection) in enumerate(executor.map(prepare_track,(root/'annotations.tsv').read_text().splitlines()[1:])):
            rows.extend(prepared)
            if rejection:
                rejected.append(rejection)
            if index % 100 == 0:
                require_resources()
                print(json.dumps(dict(preparing='jamendo',tracks=index,qa=len(rows))),flush=True)
    write_json(ROOT/'processed/music/jamendo_rejected_tracks.json',rejected)
    return rows


def prepare_nsynth(notes_per_instrument=20):
    root = ROOT/'raw/music/nsynth'
    report = json.loads((root/'provenance.json').read_text())
    if not report.get('complete'):
        raise ValueError('NSynth downloads incomplete')
    rows = []
    for split in ('train','valid','test'):
        archive = root/f'nsynth-{split}.jsonwav.tar.gz'
        destination = ROOT/'processed/music/audio/nsynth'/split
        marker = destination/'selection.json'
        if marker.exists():
            selection = json.loads(marker.read_text())
            if selection['notes_per_instrument'] != notes_per_instrument:
                raise ValueError('NSynth selection recipe changed')
            selected = selection['notes']
        else:
            require_resources()
            # Streaming extraction visits gzip once, never creates 289K WAVs.
            with tarfile.open(archive,mode='r|gz') as tar:
                metadata = None
                for member in tar:
                    if PurePosixPath(member.name).name == 'examples.json' and member.isfile():
                        metadata = json.load(tar.extractfile(member))
                        break
            if metadata is None:
                raise ValueError('NSynth examples.json not found')
            instruments = defaultdict(list)
            for name,annotation in metadata.items():
                instruments[annotation['instrument']].append(name)
            selected = {name:metadata[name] for names in instruments.values()
                        for name in sorted(names,key=lambda name:digest(['nsynth-notes-v1',name]))[:notes_per_instrument]}
            destination.mkdir(parents=True,exist_ok=True)
            found = set()
            with tarfile.open(archive,mode='r|gz') as tar:
                for member in tar:
                    name = PurePosixPath(member.name).stem
                    if not member.isfile() or PurePosixPath(member.name).suffix != '.wav' or name not in selected:
                        continue
                    temporary = destination/(name+'.wav.tmp')
                    with temporary.open('wb') as handle:
                        handle.write(tar.extractfile(member).read())
                    temporary.replace(destination/(name+'.wav'))
                    found.add(name)
            if found != set(selected):
                raise ValueError('NSynth selected WAVs incomplete')
            write_json(marker,dict(notes_per_instrument=notes_per_instrument,notes=selected,
                                  archive_sha256=sha256(archive)))
        for name,annotation in selected.items():
            original = destination/(name+'.wav')
            # Official WAV is already mono 16k; preserve source bytes.
            instrument = annotation['instrument']
            group = f'nsynth-instrument:{instrument}'
            # Official valid and test share the same 53 held-out instruments.
            # Preserve official TRAIN, then assign each held-out instrument to
            # one evaluation split so the project's three splits are disjoint.
            split_name = ('train' if split == 'train' else
                          'validation' if int(digest(['nsynth-heldout-v1',annotation.get('instrument_str',str(instrument))])[:8],16)%2 == 0 else 'test')
            base = waveform_record(original,'nsynth-note:'+name,group,
                          split_name,'NSynth','entire isolated 4s note')
            base['official_source_split'] = split
            family = annotation['instrument_family_str']
            families = ['bass','brass','flute','guitar','keyboard','mallet','organ','reed','string','synth_lead','vocal']
            rows.append(qa(base,'instrument_family','Which instrument family produces this sound?',
                           [f.replace('_',' ') for f in families],family.replace('_',' '),dict(instrument=instrument)))
            rows.append(qa(base,'instrument_source','Is the sound from an acoustic, electronic, or synthetic source?',
                           ['acoustic','electronic','synthetic'],annotation['instrument_source_str'],dict(instrument=instrument)))
    return rows


def prepare_musicbench():
    root = ROOT/'raw/music/musicbench'
    unique = {}
    for line in (root/'MusicBench_train.json').read_text().splitlines():
        item = json.loads(line)
        name = PurePosixPath(item['location']).name
        if name in unique:
            for field in ('bpm','beats','key','keyprob','chords','chords_time'):
                if item[field] != unique[name][field]:
                    raise ValueError(f'Conflicting structural labels for the same MusicBench audio: {name}')
        else:
            unique[name] = item
    # Multiple captions of the same audio are not additional independent QA.
    metadata = list(unique.values())
    protected = {}
    for line in (ROOT/'processed/v3/canonical.jsonl').open():
        old = json.loads(line)
        identity = old['source_recording_id']
        if identity.startswith('youtube:'):
            previous = protected.setdefault(identity.split(':',1)[1],old['split'])
            if previous != old['split']:
                raise ValueError('Common has inconsistent source-parent split')
    selected = {PurePosixPath(r['location']).name for r in metadata}
    destination = ROOT/'processed/music/audio/musicbench-source'
    marker = destination/'extraction.json'
    if not marker.exists():
        destination.mkdir(parents=True,exist_ok=True)
        found = set()
        with tarfile.open(root/'MusicBench.tar.gz',mode='r|gz') as tar:
            for member in tar:
                name = PurePosixPath(member.name).name
                if not member.isfile() or name not in selected:
                    continue
                if name in found:
                    raise ValueError(f'Duplicate MusicBench archive basename: {name}')
                target = destination/name
                temporary = target.with_suffix('.wav.tmp')
                with temporary.open('wb') as handle:
                    handle.write(tar.extractfile(member).read())
                temporary.replace(target)
                found.add(name)
        if found != selected:
            raise ValueError(f'Missing {len(selected-found)} MusicBench source files')
        write_json(marker,dict(files=len(found),archive_sha256=sha256(root/'MusicBench.tar.gz')))
    rows, rejected = [], Counter()
    for index,annotation in enumerate(metadata):
        name = PurePosixPath(annotation['location']).name
        parent = musicbench_parent(name)
        group = 'youtube:'+parent
        split = protected.get(parent,identity_split(group))
        source = destination/name
        info = sf.info(source)
        if not 1 <= info.duration <= 30:
            rejected['duration'] += 1
            continue
        target = ROOT/'processed/music/audio/musicbench'/name
        convert_audio(source,target)
        base = waveform_record(target,'musicbench:'+PurePosixPath(name).stem,group,split,'MusicBench',
                               'automatic labels aligned to this exact transformed clip', [0,info.duration])
        for task,question,options,answer,evidence in structure_labels(annotation,base['duration_s']):
            rows.append(qa(base,task,question,options,answer,dict(evidence,automatic=True,human_verified=False)))
        if index % 500 == 0:
            require_resources()
            print(json.dumps(dict(preparing='musicbench',clips=index,qa=len(rows))),flush=True)
    write_json(ROOT/'processed/music/musicbench_rejections.json',dict(rejected))
    return rows


def write_dataset(rows,name):
    original_count = len(rows)
    pcm_splits = defaultdict(set)
    pcm_answers = defaultdict(set)
    for row in rows:
        pcm_splits[row['pcm_identity']].add(row['split'])
        pcm_answers[row['pcm_identity'],row['question']].add(row['choices'][row['answer_index']])
    cross_split = {pcm for pcm,splits in pcm_splits.items() if len(splits)>1}
    conflicting = {key for key,answers in pcm_answers.items() if len(answers)>1}
    # Some NSynth instruments have byte-identical notes across official splits.
    # Keep the official instrument splits and exclude those recordings entirely.
    rows = [row for row in rows if row['pcm_identity'] not in cross_split and
            (row['pcm_identity'],row['question']) not in conflicting]
    groups,pcms = {},{}
    for row in rows:
        for mapping,key in ((groups,row['split_group']),(pcms,row['pcm_identity'])):
            previous = mapping.setdefault(key,row['split'])
            if previous != row['split']:
                raise ValueError(f'Music split leak: {key}')
    rows = sorted(rows,key=lambda row:digest(['music-order-v1',row['sample_id']]))
    target = ROOT/'processed/music'/name
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix('.jsonl.tmp')
    with temporary.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    temporary.replace(target)
    counters = Counter((r['split'],r['music_task']) for r in rows)
    labels = defaultdict(Counter)
    for row in rows:
        labels[row['split']+':'+row['music_task']][row['choices'][row['answer_index']]] += 1
    write_json(target.with_suffix('.report.json'),dict(samples=len(rows),input_qa=original_count,
               excluded_qa=original_count-len(rows),cross_split_duplicate_pcm=len(cross_split),
               conflicting_pcm_questions=len(conflicting),
               excluded_cross_split_pcm=sorted(cross_split),sha256=sha256(target),
               split_task_counts={':'.join(key):value for key,value in counters.items()},
               label_counts={key:dict(value) for key,value in labels.items()},
               source_groups=len(groups),unique_pcm=len(pcms),split_leaks=0,
               captions_in_state=False,label_scope_preserved=True))
    return target
