"""The official README example, with actual audio paths and data-volume caches."""
import os
from pathlib import Path

data = Path("/srv/data/samsone_clm")
os.environ.setdefault("HF_HOME", str(data / "cache/huggingface"))
os.environ.setdefault("TORCH_HOME", str(data / "cache/torch"))

from samsone import Samsone356M

model = Samsone356M()
output = model(
    audio=str(data / "cache/samsone_readme_audio/cow.wav"),
    prompt="Caption the audio",
)
print(output, flush=True)

output = model(
    audio=[str(data / "cache/samsone_readme_audio/cow.wav"),
           str(data / "cache/samsone_readme_audio/water.wav")],
    prompt="what is the difference between audio files?",
)
print(output, flush=True)
