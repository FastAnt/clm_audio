import torch

from samsone_clm.metrics import evaluate_model


def test_ablation_compares_exactly_the_same_available_samples():
    class Scores:
        def __call__(self, states, actions):
            return states

    data = {
        "correct": torch.tensor([[2., 0.], [2., 0.], [0., 2.]]),
        "random": torch.tensor([[0., 2.], [0., 0.], [0., 2.]]),
        "silence": torch.tensor([[0., 2.], [0., 0.], [0., 2.]]),
        "labels": torch.zeros(3, dtype=torch.long),
        "actions": torch.zeros(2, 2),
        "indices": torch.tensor([[0, 1], [0, 1], [0, 1]]),
        "ablation_available": torch.tensor([True, False, True]),
    }
    rows = [{"question_type": kind} for kind in ("sound", "speech", "temporal")]
    result = evaluate_model(Scores(), data, rows, torch.arange(3), batch_size=2)
    assert result["correct"]["overall"]["samples"] == 3
    assert result["ablation_correct"]["overall"]["samples"] == 2
    assert result["ablation_correct"]["overall"]["top1"] == .5
    assert result["random"]["overall"]["samples"] == 2
    assert result["audio_top1_gaps"] == {"random": .5, "silence": .5}
    assert result["random"]["speech"] == {"samples": 0}


def test_uncomputed_ablation_is_explicitly_absent():
    class Scores:
        def __call__(self, states, actions):
            return states

    data = {
        "correct": torch.tensor([[2., 0.]]),
        "random": torch.zeros(1, 2),
        "silence": torch.zeros(1, 2),
        "labels": torch.zeros(1, dtype=torch.long),
        "actions": torch.zeros(2, 2),
        "indices": torch.tensor([[0, 1]]),
        "ablation_available": torch.tensor([False]),
    }
    result = evaluate_model(Scores(), data, [{"question_type": "sound"}], torch.arange(1))
    assert result["audio_top1_gaps"] == {}
    assert result["random"]["overall"] == {"samples": 0}
    assert result["silence"]["overall"] == {"samples": 0}


def test_paired_views_do_not_make_a_sparse_task_eligible():
    class Scores:
        def __call__(self,states,actions):return states
    rows=[];scores=[]
    for i in range(60):
        rows.append(dict(sample_id=f'vocals:{i}',question_type='music',music_task='vocals',choices=['yes','no'],answer_index=0))
        scores.append([2.,0.])
    for i in range(30):
        for view in ['full','serving-window']:
            row=dict(sample_id=f'genre:{i}:{view}',question_type='music',music_task='genre',choices=['yes','no'],answer_index=0,music_view=view)
            if view=='serving-window':row['parent_sample_id']=f'genre:{i}:full'
            rows.append(row);scores.append([0.,2.])
    data=dict(correct=torch.tensor(scores),random=torch.tensor([[0.,2.]]*len(rows)),silence=torch.tensor([[0.,2.]]*len(rows)),
              labels=torch.zeros(len(rows),dtype=torch.long),actions=torch.zeros(2,2),indices=torch.tensor([[0,1]]*len(rows)),
              ablation_available=torch.ones(len(rows),dtype=torch.bool))
    result=evaluate_model(Scores(),data,rows,torch.arange(len(rows)))
    assert result['correct']['music_tasks']['genre']['samples']==60
    assert result['correct']['music_tasks']['genre']['independent_qa_samples']==30
    assert result['music_macro']['selection_tasks']==['vocals']
    assert result['music_macro']['top1']==1.
