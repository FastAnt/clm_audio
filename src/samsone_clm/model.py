from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, sha256
from clm.heads import make_head


class SharedAdapter(nn.Sequential):
    def __init__(self, input_width: int, output_width: int, bottleneck: int | None = None):
        width = bottleneck or input_width
        super().__init__(nn.LayerNorm(input_width), nn.Linear(input_width, width), nn.GELU(),
                         nn.Linear(width, output_width), nn.LayerNorm(output_width))


def load_original_heads(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["cfg"]
    state, action = ck["state_head"], ck["action_head"]
    input_width = state["inp.weight"].shape[1]
    projection_width = state["out.weight"].shape[0]
    if action["inp.weight"].shape[1] != input_width or action["out.weight"].shape[0] != projection_width:
        raise ValueError("CLM state/action checkpoint dimensions disagree")
    kw = dict(hidden=input_width, width=state["inp.weight"].shape[0], depth=cfg["depth"],
              proj=projection_width, activation=cfg.get("activation", "gelu"),
              layernorm=cfg.get("layernorm", False), residual=cfg.get("residual", False))
    sh, ah = make_head(**kw), make_head(**kw)
    sh.load_state_dict(state, strict=True)
    ah.load_state_dict(action, strict=True)
    return sh, ah, input_width, projection_width, cfg


class AudioCLM(nn.Module):
    def __init__(self, samsone_width: int, checkpoint=None, bottleneck=None, *,
                 adapter_kind="mlp", head_input_l2=True, score_scale=None, learnable_scale=False):
        super().__init__()
        checkpoint = checkpoint or ROOT / "models/clm/CLM_v0.1-8B.pt"
        self.state_head, self.action_head, input_width, projection_width, cfg = load_original_heads(checkpoint)
        if adapter_kind == "mlp":
            self.adapter = SharedAdapter(samsone_width, input_width, bottleneck)
        elif adapter_kind == "linear":
            self.adapter = nn.Linear(samsone_width, input_width)
        else:
            raise ValueError(f"Unknown adapter kind: {adapter_kind}")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        checkpoint_logit_scale = float(torch.as_tensor(ck["logit_scale"]))
        self._fixed_scale = min(math.exp(checkpoint_logit_scale), 100.) if score_scale is None else float(score_scale)
        if not math.isfinite(self._fixed_scale) or self._fixed_scale <= 0:
            raise ValueError("Score scale must be finite and positive")
        self.logit_scale = (nn.Parameter(torch.tensor(min(math.log(self._fixed_scale), math.log(100.)-1e-5)))
                            if learnable_scale else None)
        self.head_input_l2 = bool(head_input_l2)
        self.spec = dict(samsone_width=samsone_width, input_width=input_width,
                         projection_width=projection_width, bottleneck=bottleneck or samsone_width,
                         clm_sha256=sha256(checkpoint), clm_cfg=cfg, score="cosine", temperature=1./self._fixed_scale,
                         score_scale=self._fixed_scale, checkpoint_logit_scale=checkpoint_logit_scale,
                         head_input_l2=self.head_input_l2, adapter_kind=adapter_kind, contract_version=2,
                         head_matmul_precision="fp32", learnable_scale=bool(learnable_scale))
        self.set_stage(1)

    @classmethod
    def from_spec(cls, spec):
        # Legacy checkpoints retain their own scoring/input contract; never reinterpret v1 weights as v2.
        return cls(spec["samsone_width"], bottleneck=spec["bottleneck"],
                   adapter_kind=spec.get("adapter_kind", "mlp"),
                   head_input_l2=spec.get("head_input_l2", False),
                   score_scale=spec.get("score_scale", 1.), learnable_scale=spec.get("learnable_scale", False))

    @property
    def scale(self):
        return self.logit_scale.exp().clamp(max=100.) if self.logit_scale is not None else self._fixed_scale

    def enable_learnable_scale(self):
        if self.logit_scale is None:
            self.logit_scale = nn.Parameter(torch.tensor(min(math.log(self._fixed_scale), math.log(100.)-1e-5),
                                                        device=next(self.parameters()).device))
        self.spec['learnable_scale'] = True

    @torch.no_grad()
    def constrain_scale(self):
        # Upstream checkpoints can exceed log(100); avoid a permanently zero clamp gradient.
        if self.logit_scale is not None:
            self.logit_scale.clamp_(min=math.log(.01), max=math.log(100.)-1e-5)

    def _adapt(self, hidden):
        value = self.adapter(hidden.float())
        return F.normalize(value, dim=-1) if self.head_input_l2 else value

    def set_stage(self, stage: int) -> None:
        if stage not in (1, 2, 3, 4, 5, 6):
            raise ValueError("stage must be 1..6")
        self.adapter.requires_grad_(True)
        if self.logit_scale is not None:
            self.logit_scale.requires_grad_(stage >= 2)
        for head in (self.state_head, self.action_head):
            head.requires_grad_(stage >= 2)
            head.train(stage >= 2)

    def state(self, hidden):
        return F.normalize(self.state_head(self._adapt(hidden)), dim=-1)

    def action(self, hidden):
        return F.normalize(self.action_head(self._adapt(hidden)), dim=-1)

    def forward(self, states, actions, candidate_mask=None):
        # Both branches call this exact SAME adapter instance. Keep head forward differentiable in Stage 1.
        zs, za = self.state(states), self.action(actions)
        scores = self.scale * torch.einsum("bd,bkd->bk", zs, za)
        return scores.masked_fill(~candidate_mask, float("-inf")) if candidate_mask is not None else scores
