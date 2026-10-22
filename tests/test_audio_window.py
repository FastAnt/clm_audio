import numpy as np
from samsone_clm.server import audio_window


def test_window_pads_start_and_end_without_moving_playback_center():
    waveform = np.arange(12*16000, dtype=np.float32)
    first, info = audio_window(waveform, 1.)
    assert len(first)==160000 and info["silence_before_s"]==4.
    assert np.array_equal(first[:64000], np.zeros(64000))
    assert np.array_equal(first[64000:], waveform[:96000])
    last, info = audio_window(waveform, 11.)
    assert info["silence_after_s"]==4.
    assert np.array_equal(last[:96000], waveform[96000:])
    assert np.array_equal(last[96000:], np.zeros(64000))


def test_short_file_has_exact_ten_second_window():
    window, info = audio_window(np.ones(16000, dtype=np.float32), .5)
    assert len(window)==160000
    assert info["silence_before_s"]==4.5 and info["silence_after_s"]==4.5
    assert np.array_equal(window[72000:88000], np.ones(16000))
