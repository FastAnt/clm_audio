import numpy as np
import pytest
import soundfile as sf

from samsone_clm.audio import write_audio_atomic


def test_complete_pcm_published_atomically(tmp_path):
    path=tmp_path/'audio.wav'
    values=np.linspace(-.9,.9,1600,dtype=np.float32)
    write_audio_atomic(path,values,16000)
    restored,rate=sf.read(path,dtype='float32')
    assert rate==16000 and np.array_equal(values,restored)
    assert not list(tmp_path.glob('*.tmp'))


def test_truncated_writer_cannot_replace_existing_audio(tmp_path,monkeypatch):
    path=tmp_path/'audio.wav'
    original=np.ones(1600,dtype=np.float32)*.1
    write_audio_atomic(path,original,16000)
    real_write=sf.write
    def truncated(destination,values,rate,**kwargs):
        # Simulate the observed partial WAV; the old published file must survive.
        assert np.array_equal(sf.read(path,dtype='float32')[0],original)
        real_write(destination,values[:100],rate,**kwargs)
    monkeypatch.setattr(sf,'write',truncated)
    with pytest.raises(ValueError,match='complete source PCM'):
        write_audio_atomic(path,np.zeros(1600,dtype=np.float32),16000)
    assert np.array_equal(sf.read(path,dtype='float32')[0],original)
    assert not list(tmp_path.glob('*.tmp'))
