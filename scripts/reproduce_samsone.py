"""Run the official Samsone README API directly, without adapter or CLM.

Default: Samsone356M(), two single-file captions and one multi-clip comparison.
All downloaded weights and report artifacts stay on the configured data volume.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from samsone_clm.common import PROJECT, ROOT, sha256, write_json


def configure_release_cache():
    # Upstream uses Path.home()/'.cache/samsone' by default. Redirect the directory
    # itself, preserving the unmodified Samsone134M() constructor and download API.
    official_cache = Path.home() / ".cache/samsone"
    target = ROOT / "cache/samsone"
    target.mkdir(parents=True, exist_ok=True)
    if not official_cache.exists() and not official_cache.is_symlink():
        official_cache.parent.mkdir(parents=True, exist_ok=True)
        official_cache.symlink_to(target, target_is_directory=True)
    if official_cache.resolve() != target.resolve():
        raise RuntimeError(f"Official cache must resolve to {target}; existing path was left untouched: {official_cache}")
    # Reuse already verified release checkpoints without downloading a second copy.
    for source in (ROOT / "models").glob("samsone_*_model_bf16.ckpt"):
        cached = target / source.name
        if not cached.exists() and not cached.is_symlink():
            cached.symlink_to(source)
    return official_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("134m", "356m"), default="356m")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"))
    args = parser.parse_args()
    configure_release_cache()
    import torch
    from samsone import Samsone134M, Samsone356M

    torch.set_num_threads(2)
    directory = ROOT / (f"runs/samsone_readme_{args.model}" + (f"_{args.dtype}" if args.dtype else ""))
    audio_directory = ROOT / "cache/samsone_readme_audio"
    audio_directory.mkdir(parents=True, exist_ok=True)
    for name in ("cow", "water"):
        shutil.copyfile(PROJECT / f"vendor/samsone/assets/{name}.wav", audio_directory / f"{name}.wav")
    cow, water = str(audio_directory / "cow.wav"), str(audio_directory / "water.wav")
    print(f"Loading official Samsone{args.model.upper()}, dtype={args.dtype or 'default'}", flush=True)
    started = time.monotonic()
    options = {"dtype": getattr(torch, args.dtype)} if args.dtype else {}
    if args.model == "134m":
        model = Samsone134M(**options) if options else Samsone134M()
    else:
        model = Samsone356M(**options) if options else Samsone356M()
    checkpoint = ROOT / "cache/samsone" / model.spec.checkpoint_filename
    # 356M is already available under models/; its cache symlink can reuse that file.
    report = dict(
        model=type(model).__name__, constructor=(f"dtype=torch.{args.dtype}" if options else "default, no arguments"),
        source="https://github.com/SamsungLabs/samsone",
        imported_package=__import__("samsone").__file__, device=str(model.device), dtype=str(model.dtype),
        checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
        load_latency_s=time.monotonic() - started, cases=[],
    )
    write_json(directory / "results.json", report)
    cases = [
        dict(name="single_cow", audio=cow, prompt="Caption the audio"),
        dict(name="single_water", audio=water, prompt="Caption the audio"),
        dict(name="single_water_official_prompt", audio=water, prompt="describe"),
        dict(name="two_clips_one_prompt", audio=[cow, water], prompt="what is the difference between audio files?"),
        dict(name="official_cli_comparison_prompt", audio=[cow, water], prompt="compare audio files"),
    ]
    for case in cases:
        observed = {}

        def audio_hook(module, inputs, output):
            observed["audio_encoder_batch_size"] = int(output[0].shape[0])

        def text_hook(module, inputs, kwargs, output):
            observed.setdefault("text_prefill_batch_size", int(output.last_hidden_state.shape[0]))

        handles = [model.model.audio_encoder.register_forward_hook(audio_hook),
                   model.model.text_model.model.model.register_forward_hook(text_hook, with_kwargs=True)]
        started = time.monotonic()
        try:
            output = model(audio=case["audio"], prompt=case["prompt"])
        finally:
            for handle in handles:
                handle.remove()
        assert isinstance(output, str) and output.strip(), "Native API returned no response"
        assert observed["text_prefill_batch_size"] == 1, "List of paths must be one prompt/sample"
        expected_audio_batch = len(case["audio"]) if isinstance(case["audio"], list) else 1
        assert observed["audio_encoder_batch_size"] == expected_audio_batch
        result = dict(**case, output=output, latency_s=time.monotonic() - started, **observed)
        report["cases"].append(result)
        write_json(directory / "results.json", report)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    report["status"] = "completed"
    write_json(directory / "results.json", report)
    print(f"Report: {directory / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
