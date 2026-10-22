from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from samsone_clm.encoder import FrozenSamsone


def test_repeated_clip_features_restore_original_audio_order_and_silence_flag():
    encoder=FrozenSamsone.__new__(FrozenSamsone)
    calls=[]
    def extract(path,silence):
        calls.append((path,silence))
        return {'features':torch.tensor([[{'a':1,'b':2}[path]]]),'mask':torch.ones(1,2)}
    encoder._audio_features=extract
    with ThreadPoolExecutor(max_workers=2) as pool:
        encoder._feature_pool=pool
        result=encoder._audio_feature_batch(['a',Path('b'),Path('a'),'b','a'],silence=True)
    assert sorted(calls)==[('a',True),('b',True)]
    assert torch.cat(result['features']).flatten().tolist()==[1,2,1,2,1]
    assert torch.cat(result['mask']).shape==(5,2)
