"""Inventory musical examples and source-track overlap without loading a model."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

from samsone_clm.common import ROOT, write_json


def main():
    canonical = ROOT/'processed/v3/canonical.jsonl'
    counts = Counter()
    identities = defaultdict(set)
    source_counts = Counter()
    protected_youtube = set()
    for line in canonical.open():
        row = json.loads(line)
        recording = row['source_recording_id']
        if row['split'] in ('validation', 'test') and recording.startswith('youtube:'):
            protected_youtube.add(recording.split(':', 1)[1])
        if row['question_type'] == 'music':
            counts[row['split']] += 1
            identities[row['split']].add(row['audio_identity'])
            source_counts[row['source_dataset']] += 1
    raw = Counter()
    for path in (ROOT/'raw').glob('AudioMCQ_*.jsonl'):
        for line in path.open():
            row = json.loads(line)
            if row['question_type'] == 'music' and row.get('audio-contribution') == 'strong':
                raw[row['source_dataset']] += 1
    result = dict(scope='Metadata inventory only; external audio not downloaded or mixed into active training',
                  canonical_music_qa=dict(counts), canonical_music_audio={k: len(v) for k,v in identities.items()},
                  canonical_music_sources=dict(source_counts), original_strong_music_sources=dict(raw),
                  unavailable_music_qa_by_source={k: raw[k]-source_counts[k] for k in raw},
                  target_music_skills=['instruments', 'genre', 'mood', 'vocals', 'tempo', 'rhythm', 'key', 'harmony'])
    folder = ROOT/'raw/music_expansion'
    paths = folder/'MusicBench_train.json'
    if paths.exists():
        parents = set()
        protected_parents = set()
        entries = eligible = excluded = bpm = keys = 0
        for line in paths.open():
            # Official .json file is actually one JSON object per line.
            row = json.loads(line)
            stem = Path(row['location']).stem
            # YouTube IDs themselves can end in underscore+digits. Use the fixed
            # upstream source-ID width, then validate an optional augmentation suffix.
            if not re.fullmatch(r'[A-Za-z0-9_-]{11}(?:_\d+)?', stem):
                raise ValueError(f'Unexpected MusicBench audio ID: {stem}')
            parent = stem[:11]
            entries += 1
            parents.add(parent)
            if parent in protected_youtube:
                excluded += 1
                protected_parents.add(parent)
                continue
            eligible += 1
            value = row.get('bpm')
            bpm += isinstance(value, (int, float)) and 30 <= value <= 300
            confidence = row.get('keyprob') or [0.]
            keys += bool(row.get('key')) and max(confidence) >= .8
        result['musicbench_metadata'] = dict(entries=entries, parent_tracks=len(parents),
             excluded_entries_overlapping_audiomcq_heldout=excluded, excluded_parent_tracks=len(protected_parents),
             eligible_entries_after_source_overlap_filter=eligible, eligible_parent_tracks=len(parents-protected_parents),
             eligible_tempo_metadata=bpm, eligible_key_confidence_ge_08=keys,
             note='Eligibility is metadata-only: labels, audio availability/duration and PCM integrity still need verification. Variants of one parent must share a split.')
    path = folder/'mtg_jamendo_autotagging.tsv'
    if path.exists():
        tracks = artists = instruments = genres = moods = 0
        artist_ids = set()
        for row in csv.reader(path.open(), delimiter='\t'):
            if row[0] == 'TRACK_ID':
                continue
            tracks += 1
            artist_ids.add(row[1])
            tags = row[5:]
            instruments += any(t.startswith('instrument---') for t in tags)
            genres += any(t.startswith('genre---') for t in tags)
            moods += any(t.startswith('mood/theme---') for t in tags)
        result['jamendo_metadata'] = dict(tracks=tracks, artists=len(artist_ids),
              instrument_tagged_tracks=instruments, genre_tagged_tracks=genres, mood_tagged_tracks=moods,
              note='Track-level uploader tags; not ready-made MCQ or guaranteed clip-level labels.')
    output = ROOT/'runs/v3/music_inventory.json'
    write_json(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
