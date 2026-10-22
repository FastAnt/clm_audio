"""Compare checkpoints on the same held-out audio identities, with encoder provenance checks."""
import argparse
import json
from pathlib import Path

import torch

from samsone_clm.common import ROOT, digest, write_json
from samsone_clm.metrics import evaluate_model
from samsone_clm.model import AudioCLM
from samsone_clm.precompute import load_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    data, manifest = load_features(args.feature_manifest)
    data = {key: value.cuda() for key, value in data.items()}
    ids = torch.tensor([i for i, row in enumerate(manifest["rows"]) if row["split"] == "validation"], device="cuda")
    heldout_audio = {manifest["rows"][i]["audio_identity"] for i in ids.tolist()}
    result = dict(split="validation", feature_manifest=digest(manifest), samples=len(ids), runs={})
    for name, path in (("previous", args.previous), ("current", args.current)):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        for field in ("recipe", "samsone_sha256", "hidden_width", "dtype", "sample_rate", "max_audio_samples"):
            if ck["feature_provenance"][field] != manifest["provenance"][field]:
                raise ValueError(f"Encoder provenance mismatch: {name}, {field}")
        training_manifest = next((candidate for p in (ROOT / "features").glob("spike-*/manifest.json")
                                  if digest(candidate := json.loads(p.read_text())) == ck["feature_manifest"]), None)
        if training_manifest is None:
            raise ValueError(f"Original training manifest missing for {name}")
        train_audio = {row["audio_identity"] for row in training_manifest["rows"] if row["split"] == "train"}
        if heldout_audio & train_audio:
            raise ValueError(f"Validation audio overlaps training for {name}")
        model = AudioCLM(ck["spec"]["samsone_width"], bottleneck=ck["spec"]["bottleneck"])
        model.load_state_dict(ck["model_state"], strict=True)
        model.cuda().eval()
        result["runs"][name] = dict(checkpoint=str(path), trained_samples=ck["args"]["samples"],
                                    step=ck["step"], evaluation=evaluate_model(model, data, manifest["rows"], ids))
        del model, ck
    write_json(args.current.parent / "comparison_previous.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
