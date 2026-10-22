from collections import defaultdict
from samsone_clm.precompute import select_ablation_positions


def test_rare_genres_and_question_forms_share_recordings():
    rows = []
    for label, tracks in [('rock', 30), ('blues', 2), ('pop', 10)]:
        for track in range(tracks):
            parent = f'{label}:{track}'
            for question in range(6):
                rows.append(dict(split='validation', question_type='music', music_task='genre_fma',
                    genre_corpus='FMA', semantic_label=label, parent_sample_id=parent,
                    audio_identity=parent, sample_id=f'{parent}:q{question}',
                    choices=[label, 'other'], answer_index=0))
    cohort = select_ablation_positions(rows, ['validation'], 60)
    assert len(cohort) <= 60
    selected = defaultdict(set)
    for i in cohort:
        selected[rows[i]['parent_sample_id']].add(rows[i]['sample_id'])
    assert all(len(questions) == 6 for questions in selected.values())
    assert sum(parent.startswith('blues:') for parent in selected) == 2
    assert {rows[i]['semantic_label'] for i in cohort} == {'rock', 'blues', 'pop'}
    assert cohort == select_ablation_positions(rows, ['validation'], 60)
    assert not select_ablation_positions(rows, ['test'], 60)
