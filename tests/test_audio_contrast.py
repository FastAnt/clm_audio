import torch

from samsone_clm.contrast import supervised_audio_partners, supervised_swap_loss, sample_audio_pair_batch


def row(audio, answer, choices=("cow", "water"), question="What is audible?"):
    return dict(audio_identity=audio, pcm_identity=audio, question=question,
                choices=list(choices), answer_index=answer)


def test_supervised_negatives_require_same_question_choices_and_different_answer_audio():
    rows = [row("a", 0), row("b", 0, choices=("water", "cow")),
            row("c", 1, question="Where is it recorded?"), row("d", 1, choices=("cow", "bird")),
            row("heldout", 1)]
    partners = supervised_audio_partners(rows, [0, 1, 2, 3])
    assert partners.tolist() == [1, 0, -1, -1, -1]


def test_same_recording_or_pcm_is_not_a_valid_negative_even_with_conflicting_labels():
    a, b = row("same", 0), row("same", 1)
    c, d = row("c", 0), row("d", 1)
    c["question"] = d["question"] = "Another question"
    d["pcm_identity"] = "c"
    assert supervised_audio_partners([a,b,c,d], range(4)).tolist() == [-1,-1,-1,-1]


def test_swap_loss_keeps_original_choice_columns_when_partner_order_is_reversed():
    # Both labels are column 0, but the correct texts differ: cow vs water.
    rows = [row("a", 0), row("b", 0, choices=("water", "cow"))]
    data = dict(correct=torch.eye(2) * .02, actions=torch.eye(2),
                indices=torch.tensor([[0, 1], [1, 0]]), labels=torch.tensor([0, 0]),
                partners=supervised_audio_partners(rows, [0, 1]))

    class Ranker(torch.nn.Module):
        scale = 100.

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.))
            self.last_actions = None

        def forward(self, states, actions):
            self.last_actions = actions.detach().clone()
            return self.scale * self.weight * torch.einsum("bd,bcd->bc", states, actions)

    model = Ranker()
    paired = torch.tensor([0, 1])
    original_actions = data["actions"][data["indices"][paired]]
    own_scores = model(data["correct"][paired], original_actions)
    loss = supervised_swap_loss(model, data, paired, own_scores, margin=.05)
    torch.testing.assert_close(model.last_actions, original_actions)
    torch.testing.assert_close(loss, torch.tensor(.03))
    loss.backward()
    torch.testing.assert_close(model.weight.grad, torch.tensor(-.02))
    # Using the partner's order would instead give a spurious loss of .05.


def test_dedicated_batch_always_contains_verified_pairs_and_both_endpoints():
    partners = torch.tensor([1, 0, -1, -1])
    paired_rows = torch.tensor([0, 1])
    # Unpaired main rows and held-out rows can never enter the auxiliary batch.
    for _ in range(5):
        batch = sample_audio_pair_batch(partners, paired_rows, 32)
        assert len(batch) == 64
        assert batch.max() <= 1
        torch.testing.assert_close(batch[32:], partners[batch[:32]])


def test_dedicated_batch_rejects_missing_pairs():
    import pytest
    with pytest.raises(ValueError, match="verified pairs"):
        sample_audio_pair_batch(torch.tensor([-1]), torch.empty(0, dtype=torch.long), 32)


def test_reciprocal_answers_allow_different_distractors_but_require_verified_negatives():
    rows = [row("a", 0, choices=("cow", "water", "bird")),
            row("b", 0, choices=("water", "wind", "cow")),
            row("c", 0, choices=("wind", "cow", "piano")),
            row("heldout", 0, choices=("bird", "cow", "water"))]
    # a↔b are verified: each row explicitly labels the other's answer as a distractor.
    # c contains cow but a does not contain wind, so c cannot pair with a.
    partners = supervised_audio_partners(rows, [0, 1, 2], mode="reciprocal-answers")
    assert partners.tolist() == [1, 0, -1, -1]
    assert supervised_audio_partners(rows, [0, 1, 2]).tolist() == [-1, -1, -1, -1]


def test_music_pairs_cover_many_recordings_and_exclude_same_source_parent():
    rows=[row(str(i),i%3,choices=('bass','guitar','piano')) for i in range(120)]
    for i,item in enumerate(rows):
        item.update(sample_id=f'sample-{i}',split_group=f'parent-{i//3}')
    partners=supervised_audio_partners(rows,range(120),mode='reciprocal-diverse')
    assert torch.equal(partners,supervised_audio_partners(rows,range(120),mode='reciprocal-diverse'))
    assert len(set(partners.tolist()))>50
    for i,j in enumerate(partners.tolist()):
        assert j>=0
        assert rows[i]['answer_index']!=rows[j]['answer_index']
        assert rows[i]['split_group']!=rows[j]['split_group']
