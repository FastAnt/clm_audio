"""Check live rank(audio, question, choices) against the exact cached training contract."""
import argparse
import json
import time
from pathlib import Path

import torch

from samsone_clm.common import write_json
from samsone_clm.infer import Ranker
from samsone_clm.precompute import load_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    data, manifest = load_features(saved["args"]["feature_manifest"], dtype=torch.bfloat16)
    ranker = Ranker(args.checkpoint)
    records = []
    for category in ("music", "sound", "speech", "temporal"):
        i = next(i for i,r in enumerate(manifest["rows"]) if r["split"] == "validation" and r["question_type"] == category)
        row = manifest["rows"][i]
        start = time.monotonic()
        live = ranker.rank(row["audio_path"], row["question"], row["choices"])
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        with torch.no_grad():
            scores = ranker.model(data["correct"][i:i+1].cuda(), data["actions"][data["indices"][i:i+1]].cuda())
            cached = scores.softmax(-1)[0].cpu().tolist()
        by_choice = {r["choice"]:r["probability"] for r in live}
        error = max(abs(p-by_choice[c]) for c,p in zip(row["choices"], cached))
        assert error < .001, f"Live/cache mismatch {error} for {row['sample_id']}"
        assert abs(sum(by_choice.values())-1) < 1e-5
        records.append(dict(category=category, sample_id=row["sample_id"], audio_path=row["audio_path"],
                            question=row["question"], choices=row["choices"], answer_index=row["answer_index"],
                            live=live, cached_probabilities=cached, max_probability_error=error, latency_s=elapsed))
    report = dict(checkpoint=args.checkpoint, tested_real_audio_files=len(records),
                  encoder_frozen=all(not p.requires_grad for p in ranker.encoder.model.parameters()),
                  architecture=ranker.model.spec, cases=records,
                  passed=True, scope="Live/cache parity on four fixed validation examples; not an accuracy claim")
    write_json(args.output, report)
    print(json.dumps(dict(passed=True, cases=len(records), max_probability_error=max(r["max_probability_error"] for r in records)), indent=2))


if __name__ == "__main__":
    main()
