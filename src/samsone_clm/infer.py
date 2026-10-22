from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .encoder import FrozenSamsone
from .model import AudioCLM


class Ranker:
    def __init__(self, checkpoint, device="cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        self.device = torch.device(device)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        torch.backends.cuda.matmul.allow_tf32 = ck["spec"].get("head_matmul_precision", "tf32") == "tf32"
        self.model = AudioCLM.from_spec(ck["spec"])
        self.model.load_state_dict(ck["model_state"], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.encoder = FrozenSamsone.from_provenance(ck["feature_provenance"], device=device)
        for key, value in self.encoder.provenance.items():
            if ck["feature_provenance"][key] != value:
                raise ValueError(f"Encoder mismatch: {key}")
        self.action_cache = {}
        self.checkpoint = str(checkpoint)

    def reload(self, checkpoint):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        torch.backends.cuda.matmul.allow_tf32 = ck["spec"].get("head_matmul_precision", "tf32") == "tf32"
        new_encoder = self.encoder
        if any(ck["feature_provenance"].get(key) != value for key,value in self.encoder.provenance.items()):
            new_encoder = FrozenSamsone.from_provenance(ck["feature_provenance"], device=str(self.device))
        for key, value in new_encoder.provenance.items():
            if ck["feature_provenance"][key] != value:
                raise ValueError(f"Encoder mismatch during checkpoint reload: {key}")
        model = AudioCLM.from_spec(ck["spec"])
        model.load_state_dict(ck["model_state"], strict=True)
        self.model = model.to(self.device).eval().requires_grad_(False)
        if new_encoder is not self.encoder:
            self.encoder._feature_pool.shutdown(wait=True)
            self.encoder = new_encoder
        self.action_cache.clear()
        self.checkpoint = str(checkpoint)

    @torch.no_grad()
    def rank(self, audio, question, choices):
        if len(choices) < 2 or any(not c.strip() for c in choices) or len(set(choices)) != len(choices):
            raise ValueError("Provide at least two distinct nonempty choices")
        missing = [c for c in choices if c not in self.action_cache]
        if missing:
            hidden = self.encoder.action(missing).to(torch.bfloat16).to(self.device)
            vectors = self.model.action(hidden)
            self.action_cache.update(zip(missing, vectors.unbind()))
        state = self.encoder.state([audio], [question]).to(torch.bfloat16).to(self.device)
        scores = self.model.scale * (self.model.state(state)[0] @ torch.stack([self.action_cache[c] for c in choices]).T)
        probs = scores.softmax(-1).cpu().tolist()
        return sorted([dict(choice=c, probability=p) for c, p in zip(choices, probs)], key=lambda item: -item["probability"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--choices", nargs="+", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    ranker = Ranker(args.checkpoint)
    started = time.monotonic()
    result = ranker.rank(args.audio, args.question, args.choices)
    torch.cuda.synchronize()
    print(json.dumps(dict(ranked=result, latency_s=time.monotonic()-started, checkpoint=args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
