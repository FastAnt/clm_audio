"""Compare production batched hidden extraction against official native prefill."""
import argparse
import json

import torch
from torch.nn import functional as F

from samsone_clm.common import ROOT, write_json
from samsone_clm.encoder import FrozenSamsone


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = ROOT / f"runs/samsone_native_audit_{args.device}"
    cases = json.loads((directory / "native_cases.json").read_text())["cases"]
    encoder = FrozenSamsone(device=args.device)
    from samsone.data.constants import AUDIO_TO_SAMPLE_ASSIGNMENT_KEY

    reports = []
    for batch_size in (2, len(cases)):
        batch = cases[:batch_size]
        paths = [row["audio_path"] for row in batch]
        questions = [row["question"] for row in batch]
        captured = {}

        def capture(module, inputs, kwargs, output):
            if "hidden" not in captured:
                captured["hidden"] = output.last_hidden_state.detach().cpu().clone()
                captured["mask"] = kwargs["attention_mask"].detach().cpu().clone()
                captured["embeds"] = kwargs["inputs_embeds"].detach().clone()
                captured["position_ids"] = kwargs["position_ids"].detach().cpu().clone()

        hook = encoder.transformer.register_forward_hook(capture, with_kwargs=True)
        pre = encoder.helper._tokenize("")
        pre = {key: value.expand(batch_size, -1) for key, value in pre.items()}
        try:
            # Official AudioLM.generate supports multiple independent samples via
            # the assignment mapping. The helper's list-of-paths API is ONE sample.
            encoder.model.generate(
                audio_features_batch=encoder.helper._prepare_audio(paths),
                metadata_batch={AUDIO_TO_SAMPLE_ASSIGNMENT_KEY: torch.arange(batch_size, device=encoder.device)},
                pre_audio_tokens_batch=pre,
                post_audio_tokens_batch=encoder._tokens([q + " answer: " for q in questions]),
                max_new_tokens=1,
            )
        finally:
            hook.remove()
        native = encoder._last_valid(captured["hidden"], captured["mask"]).float()
        replay = encoder._forward(captured["embeds"], captured["mask"].to(encoder.device)).float()
        official_positions_out = encoder.transformer(
            inputs_embeds=captured["embeds"],
            attention_mask=captured["mask"].to(encoder.device),
            position_ids=captured["position_ids"].to(encoder.device),
            use_cache=False,
            return_dict=True,
        )
        official_positions = encoder._last_valid(
            official_positions_out.last_hidden_state,
            captured["mask"].to(encoder.device),
        ).float().cpu()
        production = encoder.state(paths, questions).float()
        result = {"batch_size": batch_size, "valid_tokens": captured["mask"].sum(-1).tolist()}
        for name, value in (("same_embeds_replay", replay), ("production", production),
                            ("official_position_ids_replay", official_positions)):
            result[name] = dict(
                max_abs_error=float((native - value).abs().max()),
                mean_relative_l2_error=float(((native - value).norm(dim=-1) / native.norm(dim=-1)).mean()),
                mean_cosine=float(F.cosine_similarity(native, value, dim=-1).mean()),
            )
        reports.append(result)
        print(json.dumps(result), flush=True)
        write_json(directory / "batch_prefill_proof.json", {"device": args.device, "batches": reports})


if __name__ == "__main__":
    main()
