import argparse
import json
from pathlib import Path

import torch

from .common import write_json
from .metrics import evaluate_model
from .model import AudioCLM
from .precompute import load_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    ap.add_argument("--feature-manifest")
    ap.add_argument("--output",type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    data, manifest = load_features(args.feature_manifest, dtype=torch.bfloat16)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    torch.backends.cuda.matmul.allow_tf32 = ck["spec"].get("head_matmul_precision", "tf32") == "tf32"
    if ck["feature_provenance"] != manifest["provenance"]:
        raise ValueError("Checkpoint/features provenance mismatch")
    model = AudioCLM.from_spec(ck["spec"])
    model.load_state_dict(ck["model_state"])
    model.cuda().eval()
    data = {k: v.cuda() for k, v in data.items()}
    ids = torch.tensor([i for i, r in enumerate(manifest["rows"]) if r["split"] == args.split], device="cuda")
    if not len(ids):
        raise ValueError("Empty evaluation split")
    result = evaluate_model(model, data, manifest["rows"], ids)
    write_json(args.output or Path(args.checkpoint).parent / f"evaluation_{args.split}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
