import torch
from safetensors.torch import save_file

from samsone_clm.common import digest,sha256,write_json
from samsone_clm.precompute import load_features
from samsone_clm.metrics import evaluate_model
from samsone_clm.metrics import music_task_metrics


def test_variable_candidates_never_give_padding_probability_and_preserve_manifest(tmp_path):
    rows = [dict(answer_index=0,choices=['a','b'],question_type='music',music_task='vocals'),
            dict(answer_index=2,choices=['a','b','c'],question_type='music',music_task='key')]
    scores = torch.tensor([[3.,1.,100.],[1.,0.,3.]])
    for mode in ('correct','random','silence','actions'):
        hidden = scores if mode != 'actions' else torch.eye(3)
        save_file({'hidden':hidden},str(tmp_path/f'{mode}-000000.safetensors'))
    manifest = dict(directory=str(tmp_path),rows=rows,choices=['a','b','c'],
                    candidate_indices=[[0,1],[0,1,2]],provenance=dict(hidden_width=3),
                    shard_sha256={p.name:sha256(p) for p in tmp_path.glob('*.safetensors')})
    write_json(tmp_path/'manifest.json',manifest)
    data,stored = load_features(tmp_path/'manifest.json')
    assert digest(stored) == digest(manifest)
    assert data['candidate_mask'].tolist() == [[True,True,False],[True,True,True]]

    class Scores:
        def __call__(self,states,actions):
            return states

    report = evaluate_model(Scores(),data,rows,torch.arange(2))
    assert report['correct']['overall']['top1'] == 1.
    assert report['correct']['music_tasks']['vocals']['mean_correct_probability'] > .8
    assert report['music_macro']['top1'] == 1.
    assert report['music_macro']['audio_top1_gaps'] == {'random':0.,'silence':0.}


def test_class_macro_uses_answer_texts_with_shuffled_choice_columns():
    rows=[dict(choices=choices,answer_index=label) for choices,label in
          [(['voice','instrumental'],0),(['instrumental','voice'],1),
           (['voice','instrumental'],0),(['instrumental','voice'],0)]]
    labels=torch.tensor([0,1,0,0])
    always_voice=torch.tensor([[10.,0.],[0.,10.],[10.,0.],[0.,10.]])
    report=music_task_metrics(always_voice,labels,rows)
    assert report['top1']==.75
    assert report['class_macro_top1']==.5
    assert report['semantic_class_counts']=={'instrumental':1,'voice':3}
