"""Audio decoding without importing PyTorch or allocating an encoder."""
import subprocess
import os
import uuid
from pathlib import Path
import numpy as np


def write_audio_atomic(path, waveform, sample_rate):
    """Publish a WAV only after an exact PCM roundtrip and a durable complete write."""
    import soundfile as sf
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        sf.write(temporary,waveform,sample_rate,subtype='FLOAT',format='WAV')
        restored,rate=sf.read(temporary,dtype='float32')
        if rate!=sample_rate or restored.shape!=waveform.shape or not np.array_equal(restored,waveform):
            raise ValueError('Written WAV differs from complete source PCM')
        with temporary.open('rb') as handle:os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def decode_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    from .common import resolve_data_path
    path = resolve_data_path(path)
    # Prepared mono WAV already has the encoder's sample rate. Reading its
    # samples directly avoids a fresh ffmpeg process per QA. Restrict the path
    # to encodings verified bit-for-bit against the existing decoder.
    if path.suffix.lower()=='.wav':
        import soundfile as sf
        try:
            info=sf.info(path)
            if info.format in ('WAV','WAVEX') and info.channels==1 and info.samplerate==sample_rate and info.subtype in ('FLOAT','PCM_16'):
                audio,rate=sf.read(path,dtype='float32')
                if rate!=sample_rate or not len(audio) or not np.isfinite(audio).all():
                    raise ValueError(f'Invalid decoded audio: {path}')
                return audio
        except sf.LibsndfileError:
            pass  # Preserve ffmpeg support for containers libsndfile cannot open.
    output = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", "-i", str(path),
                             "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "-"],
                            check=True, capture_output=True).stdout
    audio = np.frombuffer(output, dtype="<f4").copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid decoded audio: {path}")
    return audio
