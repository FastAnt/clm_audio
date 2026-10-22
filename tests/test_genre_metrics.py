import pytest
import torch
from samsone_clm.metrics import music_task_metrics,genre_selection_key

def test_answer_aliases_count_as_one_semantic_class():
    rows=[dict(choices=[text,'other'],answer_index=0,semantic_label='pop',sample_id=str(i))
          for i,text in enumerate(['pop','pop music','popular music'])]
    rows += [dict(choices=['hip hop','other'],answer_index=0,semantic_label='hip hop',sample_id='hip')]
    scores=torch.tensor([[2.,0.],[0.,2.],[0.,2.],[2.,0.]])
    value=music_task_metrics(scores,torch.zeros(4,dtype=torch.long),rows)
    assert value['semantic_class_counts']=={'hip hop':1,'pop':3}
    assert value['class_macro_top1']==pytest.approx(2/3)

def fixture(mean,worst,structure):
    return {'genre_evaluation':{'music_macro':{'top1':mean,'loss':1.}},
            'genre_question_variants':{'short':{'music_macro':{'top1':worst,'audio_top1_gaps':{'random':.1,'silence':.1}}},
                                       'template':{'music_macro':{'top1':mean,'audio_top1_gaps':{'random':.1,'silence':.1}}}},
            'replay_evaluation':{'music_macro':{'family_top1':{'perception':.8,'structure':structure}}}}

def test_genre_improvement_must_retain_replay_and_worst_question():
    initial=fixture(.4,.3,.45)
    assert genre_selection_key(fixture(.6,.5,.45),initial)[0]==1
    assert genre_selection_key(fixture(.6,.5,.40),initial)[0]==0
    assert genre_selection_key(fixture(.6,.29,.45),initial)[0]==0
