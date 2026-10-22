import json
import argparse
from pathlib import Path

import torch

from samsone_clm.common import ROOT, write_json
from samsone_clm.infer import Ranker
from samsone_clm.precompute import load_features


def main():
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/overfit-256-seed42-lr0.001/best.pt")
    parser.add_argument("--feature-manifest")
    parser.add_argument("--probability-atol", type=float, default=.01,
                        help="Absolute probability tolerance for BF16 encoder outputs with different padded batch shapes")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    ranker = Ranker(checkpoint)
    data, manifest = load_features(args.feature_manifest)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["feature_provenance"] != manifest["provenance"]:
        raise ValueError("Checkpoint/features provenance mismatch")
    del saved
    index = next(i for i, r in enumerate(manifest["rows"]) if r["split"] == "train")
    row = manifest["rows"][index]
    ranked = ranker.rank(row["audio_path"], row["question"], row["choices"])
    with torch.no_grad():
        reference = ranker.model(data["correct"][index:index+1].cuda(),
                                 data["actions"][data["indices"][index:index+1]].cuda()).softmax(-1)[0].cpu().tolist()
    by_choice = {item["choice"]: item["probability"] for item in ranked}
    error = max(abs(by_choice[c]-p) for c, p in zip(row["choices"], reference))
    if error > args.probability_atol:
        raise ValueError(f"Live/cached probabilities differ by {error:.6f}; BF16 tolerance {args.probability_atol}")
    assert abs(sum(by_choice.values())-1) < 1e-5
    unseen = ["piano", "guitar", "violin", "traffic", "water"]
    new_candidates = ranker.rank(row["audio_path"], "what sound is heard?", unseen)
    assert len(new_candidates) == len(unseen) and abs(sum(r["probability"] for r in new_candidates)-1) < 1e-5
    report = dict(checkpoint=str(checkpoint), audio=row["audio_path"], question=row["question"],
                  ranked=ranked, cached_feature_probability_max_abs_error=error,
                  probability_absolute_tolerance=args.probability_atol,
                  cached_live_top1_agree=ranked[0]["choice"] == row["choices"][max(range(len(reference)), key=reference.__getitem__)],
                  predicted_correct=ranked[0]["choice"] == row["choices"][row["answer_index"]],
                  arbitrary_new_candidates=new_candidates, samsone_frozen=all(not p.requires_grad for p in ranker.encoder.model.parameters()))
    write_json(checkpoint.parent / "inference_proof.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
