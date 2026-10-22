import subprocess

import numpy as np
import pytest
import soundfile as sf

from samsone_clm.audio import decode_audio


@pytest.mark.parametrize('subtype',['FLOAT','PCM_16'])
@pytest.mark.parametrize('container',['WAV','WAVEX'])
def test_prepared_mono_wav_fast_read_is_bit_exact_with_ffmpeg(tmp_path,subtype,container,monkeypatch):
    path=tmp_path/'audio.wav'
    samples=np.random.default_rng(123).uniform(-.99,.99,16000).astype('float32')
    sf.write(path,samples,16000,subtype=subtype,format=container)
    reference=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-filter_threads','1',
              '-i',str(path),'-f','f32le','-acodec','pcm_f32le','-ac','1','-ar','16000','-'],
              check=True,capture_output=True).stdout
    def unexpected_process(*args,**kwargs):raise AssertionError('Prepared WAV must not spawn ffmpeg')
    monkeypatch.setattr(subprocess,'run',unexpected_process)
    assert np.array_equal(decode_audio(path),np.frombuffer(reference,dtype='<f4'))
