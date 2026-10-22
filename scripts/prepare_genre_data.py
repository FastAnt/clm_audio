"""Official FMA primary genre QA, artist-disjoint, exact demo windows and paraphrases."""
import argparse,csv,hashlib,json,os,zlib
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zipfile
import numpy as np
import soundfile as sf
from samsone_clm.audio import decode_audio
from samsone_clm.common import ROOT,digest,sha256,write_json
from samsone_clm.music_data import waveform_record,qa
from samsone_clm.resources import require_resources
from samsone_clm.server import audio_window

LABELS={'Blues':'blues','Classical':'classical','Country':'country','Electronic':'electronic',
        'Experimental':'experimental','Folk':'folk','Hip-Hop':'hip hop','Jazz':'jazz',
        'Pop':'pop','Rock':'rock','Soul-RnB':'soul rnb'}
ALIASES={'blues':'blues music','classical':'classical music','country':'country music',
         'electronic':'electronic music','experimental':'experimental music','folk':'folk music',
         'hip hop':'rap hiphop','jazz':'jazz music','pop':'pop music','rock':'rock music',
         'soul rnb':'rhythm and blues'}
QUESTIONS=['Which broad musical genre best describes this excerpt?',
           'what is music genre?','What genre is this music?','What style of music is playing?',
           'Which musical genre is playing?','What type of music is heard?']

def metadata():
    root=ROOT/'raw/music/fma'
    with zipfile.ZipFile(root/'fma_metadata.zip') as z:
        for name in ['tracks.csv','genres.csv']:
            path=root/name
            if not path.exists():path.write_bytes(z.read('fma_metadata/'+name))
    with (root/'tracks.csv').open(newline='') as f:
        reader=csv.reader(f);keys=list(zip(next(reader),next(reader)));next(reader)
        for fields in reader:
            d=dict(zip(keys,fields))
            yield dict(id=fields[0],artist=d[('artist','id')],genre=d[('track','genre_top')],
                       split={'training':'train','validation':'validation','test':'test'}[d[('set','split')]],
                       subset=d[('set','subset')],license=d[('track','license')])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--subset',choices=['medium','large'],default='medium')
    ap.add_argument('--workers',type=int,default=8);args=ap.parse_args();require_resources()
    raw=ROOT/'raw/music/fma';name='fma_'+args.subset+'.zip'
    verification=json.loads((raw/(name+'.verified.json')).read_text())
    assert verification['verified'] and (raw/name).stat().st_size==verification['bytes']
    tracks=[r for r in metadata() if r['genre'] in LABELS and (args.subset=='large' or r['subset'] in ('small','medium'))]
    artists=defaultdict(set)
    for r in tracks:artists[r['artist'] or ('track:'+r['id'])].add(r['split'])
    if any(len(x)>1 for x in artists.values()):raise ValueError('Official FMA artist splits overlap')
    known={}
    for dataset in ['processed/full.jsonl','processed/music/nsynth.jsonl','processed/music/jamendo.jsonl','processed/music/musicbench.jsonl']:
        path=ROOT/dataset
        if path.exists():
            with path.open() as handle:
                for line in handle:
                    row=json.loads(line)
                    for key in ['pcm_identity','parent_pcm_identity']:
                        if row.get(key):known.setdefault(row[key],set()).add(row['split'])
    out=ROOT/'processed/music_genre';out.mkdir(parents=True,exist_ok=True)
    archive=zipfile.ZipFile(raw/name)
    def prepare(row):
        track=row['id'];padded=track.zfill(6);entry='fma_'+args.subset+'/'+padded[:3]+'/'+padded+'.mp3'
        source=raw/'audio'/(padded+'.mp3');source.parent.mkdir(parents=True,exist_ok=True)
        try:
            if not source.exists():
                temp=source.with_suffix('.mp3.tmp')
                temp.write_bytes(archive.read(entry));temp.replace(source)
            if zlib.crc32(source.read_bytes())!=archive.getinfo(entry).CRC:
                return dict(track=track,error='Source differs from verified archive entry CRC'),[]
            audio=decode_audio(source)
            if not np.isfinite(audio).all() or len(audio)<16000 or np.sqrt(np.mean(audio.astype('float64')**2))<1e-6:
                return dict(track=track,error='Invalid or silent audio'),[]
            parent='pcm:'+digest([16000,hashlib.sha256(audio.tobytes()).hexdigest()])
            if parent in known:
                return dict(track=track,error='Existing corpus PCM overlap',protected_splits=sorted(known[parent])),[]
            rows=[]
            positions=[7.5,22.5] if row['split']=='train' else [len(audio)/32000]
            for wi,position in enumerate(positions):
                window,details=audio_window(audio,position)
                target=out/'audio'/row['split']/(padded+'-'+str(wi)+'.wav');target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    temp=target.with_suffix('.wav.tmp');sf.write(temp,window,16000,subtype='FLOAT',format='WAV');temp.replace(target)
                base=waveform_record(target,'fma-track:'+track,'fma-artist:'+(row['artist'] or ('track:'+track)),row['split'],
                                     'FMA official primary genre',
                                     'Primary track genre on released 30s excerpt projected to 10s; weak local labels',
                                     [details['source_start_s'],details['source_end_s']])
                base.update(parent_pcm_identity=parent,source_audio_sha256=sha256(source),license=row['license'],
                            music_view='serving-window',genre_corpus='FMA')
                for qi,question in enumerate(QUESTIONS[:4] if row['split']=='train' else QUESTIONS):
                    alias=bool(qi%2) if row['split']=='train' else False
                    choices=[ALIASES[label] if alias else label for label in LABELS.values()]
                    label=LABELS[row['genre']];answer=ALIASES[label] if alias else label
                    item=qa(base,'genre_fma',question,choices,answer,
                            dict(official_primary_genre=row['genre'],archive_sha256=verification['sha256'],
                                 source_track_id=track,label_type='official genre metadata, weak excerpt projection'))
                    item.update(sample_id=digest(['fma-genre-v1',track,wi,qi,alias]),
                                parent_sample_id='fma-genre-primary:'+track,semantic_label=label,
                                question_variant_id=str(qi),choice_variant_id=int(alias),
                                unseen_question_form=qi>=4)
                    rows.append(item)
            return dict(track=track,parent_pcm=parent,label=LABELS[row['genre']],split=row['split']),rows
        except Exception as e:return dict(track=track,error=type(e).__name__+': '+str(e)),[]
    records=[];rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,(record,items) in enumerate(pool.map(prepare,tracks)):
            records.append(record);rows.extend(items)
            if i%256==0:
                require_resources();write_json(out/('fma_'+args.subset+'.progress.json'),dict(done=i+1,total=len(tracks),qa=len(rows)))
                print(f'FMA prepare {i+1}/{len(tracks)}, {len(rows)} QA',flush=True)
    archive.close()
    labels=defaultdict(set);splits=defaultdict(set)
    for r in records:
        if r.get('parent_pcm'):labels[r['parent_pcm']].add(r['label']);splits[r['parent_pcm']].add(r['split'])
    bad={pcm for pcm,values in labels.items() if len(values)>1 or len(splits[pcm])>1}
    rows=[r for r in rows if r['parent_pcm_identity'] not in bad]
    # Exact window PCM split duplicates are excluded, not reassigned silently.
    seen=defaultdict(set)
    for r in rows:seen[r['pcm_identity']].add(r['split'])
    bad_windows={key for key,value in seen.items() if len(value)>1}
    rows=[r for r in rows if r['pcm_identity'] not in bad_windows]
    rows.sort(key=lambda r:r['sample_id'])
    path=out/('fma_'+args.subset+'.jsonl');temp=path.with_suffix('.jsonl.tmp')
    with temp.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    temp.replace(path)
    report=dict(dataset='FMA '+args.subset,samples=len(rows),unique_recordings=len({r['audio_identity'] for r in rows}),
                unique_train_recordings=len({r['audio_identity'] for r in rows if r['split']=='train'}),
                split_counts=dict(Counter(r['split'] for r in rows)),
                genre_train_qa=dict(Counter(r['semantic_label'] for r in rows if r['split']=='train')),
                split_leaks=0,excluded_source_pcm_groups=len(bad),excluded_window_pcm_groups=len(bad_windows),
                errors=[r for r in records if 'error' in r],sha256=sha256(path),
                architecture='Frozen Samsone356M, same shared adapter and original CLM heads',
                training_questions=4,heldout_questions=6,unseen_question_ids=['4','5'],
                file_sha256_verified=True,official_archive=verification)
    write_json(path.with_suffix('.report.json'),report)
    print(json.dumps({k:v for k,v in report.items() if k!='errors'}),flush=True)

if __name__=='__main__':main()
