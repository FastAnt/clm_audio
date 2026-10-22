import json
import subprocess

import torch
from torch.nn import functional as F

from samsone_clm.common import PROJECT, ROOT, read_rows, write_json
from samsone_clm.encoder import FrozenSamsone
from samsone_clm.model import AudioCLM


def main():
    torch.set_num_threads(8)
    torch.manual_seed(42)
    row, other = read_rows(ROOT / "processed/spike.jsonl")[:2]
    encoder = FrozenSamsone()
    correct = encoder.state([row["audio_path"]], [row["question"]])
    repeat = encoder.state([row["audio_path"]], [row["question"]])
    silence = encoder.state([row["audio_path"]], [row["question"]], silence=True)
    changed_question = encoder.state([row["audio_path"]], [other["question"]])
    action = encoder.action(row["choices"])
    batched = encoder.action([row["choices"][0], "piano"])[0]
    single = encoder.action([row["choices"][0]])[0]
    model = AudioCLM(encoder.width).cuda()
    scores = model(correct.cuda(), action.cuda().unsqueeze(0))
    F.cross_entropy(scores, torch.tensor([row["answer_index"]], device="cuda")).backward()
    adapter_gradient = sum(p.grad.float().norm().item() for p in model.adapter.parameters() if p.grad is not None)
    report = dict(provenance=encoder.provenance, model=model.spec,
                  state_shape=list(correct.shape), action_shape=list(action.shape),
                  deterministic=torch.equal(correct, repeat),
                  changed_audio_l2=(correct.float()-silence.float()).norm().item(),
                  changed_question_l2=(correct.float()-changed_question.float()).norm().item(),
                  action_batch_max_abs=(batched.float()-single.float()).abs().max().item(),
                  frozen_samsone=all(not p.requires_grad for p in encoder.model.parameters()),
                  frozen_heads=all(not p.requires_grad for head in [model.state_head, model.action_head] for p in head.parameters()),
                  adapter_gradient_norm_sum=adapter_gradient,
                  scores=scores.detach().cpu().tolist(), audio_file=row["audio_path"], sample_id=row["sample_id"],
                  git_revisions={repo: subprocess.check_output(["git", "-C", str(PROJECT/"vendor"/repo), "rev-parse", "HEAD"], text=True).strip()
                                 for repo in ["samsone", "CLM"]}, torch_version=str(torch.__version__))
    assert report["deterministic"] and report["frozen_samsone"] and report["frozen_heads"]
    assert report["changed_audio_l2"] > 0 and report["changed_question_l2"] > 0 and adapter_gradient > 0
    write_json(ROOT / "runs/extraction_proof.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
