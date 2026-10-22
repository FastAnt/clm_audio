"""Check native batched prefill, batch stability and BF16 storage boundary on real audio."""
import json

import torch
from torch.nn import functional as F

from samsone_clm.common import ROOT, write_json
from samsone_clm.encoder import FrozenSamsone


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    cases = json.loads((ROOT / "runs/samsone_native_audit_cuda/native_cases.json").read_text())["cases"]
    encoder = FrozenSamsone()
    batch = cases[:4]
    paths, questions = [c["audio_path"] for c in batch], [c["question"] for c in batch]
    captured = {}
    def hook(module, inputs, kwargs, output):
        if not captured:
            captured.update(hidden=output.last_hidden_state.cpu().clone(),
                            mask=kwargs["attention_mask"].cpu().clone())
    handle = encoder.transformer.register_forward_hook(hook, with_kwargs=True)
    pre = {key:value.expand(len(batch), -1) for key,value in encoder.helper._tokenize("").items()}
    from samsone.data.constants import AUDIO_TO_SAMPLE_ASSIGNMENT_KEY
    try:
        encoder.model.generate(
            audio_features_batch=encoder.helper._prepare_audio(paths),
            metadata_batch={AUDIO_TO_SAMPLE_ASSIGNMENT_KEY:torch.arange(len(batch),device=encoder.device)},
            pre_audio_tokens_batch=pre,
            post_audio_tokens_batch=encoder._tokens([q+" answer: " for q in questions]), max_new_tokens=1)
    finally:
        handle.remove()
    native = encoder._last_valid(captured["hidden"],captured["mask"])
    production = encoder.state(paths,questions)
    singles = torch.cat([encoder.state([path],[question]) for path,question in zip(paths,questions)])
    choices = ["piano", "electric guitar and drums", "train horn", "the speaker is neutral"]
    action_single = torch.cat([encoder.action([choice]) for choice in choices])
    action_batch = encoder.action(choices)
    action_large = encoder.action(choices * 32)[:4]
    def compare(a,b):
        return dict(max_abs_error=float((a.float()-b.float()).abs().max()),
                    mean_relative_l2=float(((a.float()-b.float()).norm(dim=-1)/a.float().norm(dim=-1)).mean()))
    report = dict(provenance=encoder.provenance, frozen=all(not p.requires_grad for p in encoder.model.parameters()),
                  native_vs_production=compare(native,production), state_single_vs_batch=compare(singles,production),
                  action_single_vs_4=compare(action_single,action_batch), action_4_vs_128=compare(action_batch,action_large),
                  stored_state_single_vs_batch=compare(singles.bfloat16(),production.bfloat16()),
                  stored_action_4_vs_128=compare(action_batch.bfloat16(),action_large.bfloat16()))
    assert report["frozen"]
    assert report["native_vs_production"]["mean_relative_l2"] < 1e-5
    assert report["state_single_vs_batch"]["mean_relative_l2"] < 1e-4
    assert report["action_4_vs_128"]["mean_relative_l2"] < 1e-4
    write_json(ROOT / "runs/v2/encoder_proof.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
