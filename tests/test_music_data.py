import pytest

from samsone_clm.music_data import musicbench_parent, normalize_pitch, structure_labels


def test_musicbench_parent_preserves_youtube_ids_ending_in_digits():
    original = 'abcdEFGHI_1'
    assert len(original) == 11
    assert musicbench_parent('data_aug2/'+original+'.wav') == original
    assert musicbench_parent(original+'_12.wav') == original
    with pytest.raises(ValueError,match='source identity'):
        musicbench_parent('unexpected-short.wav')


def test_structure_labels_require_consistent_beats_and_key_confidence():
    item = dict(bpm=100,beats=[[i*.6 for i in range(16)],[i%4+1 for i in range(16)]],
                key=['Db','major'],keyprob=[.9],main_caption='a piano in C minor',chords=['C'])
    labels = {task:(choices,answer,evidence) for task,question,choices,answer,evidence in structure_labels(item,10)}
    assert labels['tempo'][1] == '80 to 119 BPM'
    assert labels['meter'][1] == 'four beats per bar'
    assert labels['key_tonic'][1] == 'C sharp'
    assert labels['key_mode'][1] == 'major'
    assert 'chord' not in labels
    item['bpm'] = 200  # Half/double mismatch is not treated as verified tempo.
    item['keyprob'] = [.7]
    assert [label[0] for label in structure_labels(item,10)] == ['meter']
    item['beats'][0][-1] = 11  # Annotation lies outside this exact transformed clip.
    assert structure_labels(item,10) == []


def test_pitch_labels_remain_ascii_and_equivalent_enharmonic_notes_match():
    assert normalize_pitch('F#') == normalize_pitch('Gb') == 'F sharp'
    assert normalize_pitch('A♭') == 'A flat'
    assert normalize_pitch('not a note') is None
