from __future__ import annotations

import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, digest, sha256
from .audio import decode_audio


class FrozenSamsone:
    legacy_recipe = "samsone356m-final-last-valid-v1-state-answer-suffix-action-text-only"
    recipe = "samsone356m-final-last-valid-v2-native-positions-fp32-state-answer-suffix-action-text-only"

    def __init__(self, checkpoint=None, device="cuda", *, recipe=None, compute_dtype=None):
        # Set explicit cache before constructing upstream inference helper (including prune map).
        import samsone.inference.download as download
        download.CACHE_DIR = ROOT / "cache/samsone"
        import samsone.inference.models as upstream
        upstream.CACHE_DIR = download.CACHE_DIR
        self.checkpoint = Path(checkpoint or ROOT / "models/samsone_356m_model_bf16.ckpt")
        self.recipe = recipe or type(self).recipe
        if self.recipe not in (type(self).recipe, self.legacy_recipe):
            raise ValueError(f"Unsupported encoder recipe: {self.recipe}")
        self.legacy = self.recipe == self.legacy_recipe
        if not self.legacy:
            torch.backends.cuda.matmul.allow_tf32 = False
        self.compute_dtype = compute_dtype or (torch.bfloat16 if self.legacy else torch.float32)
        self.helper = upstream.Samsone356M(checkpoint_path=self.checkpoint, device=device, dtype=self.compute_dtype)
        self.model = self.helper.model.eval().requires_grad_(False)
        self.transformer = self.model.text_model.model.model
        self.width = self.transformer.config.hidden_size
        self.device = self.helper.device
        self._feature_pool = ThreadPoolExecutor(max_workers=8)
        self.provenance = dict(recipe=self.recipe, samsone_sha256=sha256(self.checkpoint),
                               hidden_width=self.width, dtype="bfloat16", sample_rate=self.helper.sample_rate,
                               max_audio_samples=self.helper.max_audio_samples)
        if not self.legacy:
            self.provenance.update(compute_dtype=str(self.compute_dtype).removeprefix("torch."),
                                   position_recipe="native-attention-mask-cumsum-v1")

    @classmethod
    def from_provenance(cls, provenance, device="cuda"):
        dtype = getattr(torch, provenance.get("compute_dtype", provenance["dtype"]))
        return cls(device=device, recipe=provenance["recipe"], compute_dtype=dtype)

    def _tokens(self, texts):
        normalized = [self.helper._normalise_prompt(text) for text in texts]
        if any(not text.strip() for text in normalized):
            raise ValueError("Empty/non-ASCII-only text after official Samsone normalization")
        tok = self.model.text_model.tokenizer(normalized, return_tensors="pt", padding=True)
        mapping = self.helper._original_to_pruned_token_id
        ids = tok["input_ids"]
        try:
            mapped = [[mapping[int(i)] for i in row] for row in ids]
        except KeyError as error:
            raise ValueError(f"Unavailable token in pruned Samsone vocabulary: {error}") from error
        return {"input_ids": torch.tensor(mapped, device=self.device),
                "attention_mask": tok["attention_mask"].to(self.device)}

    @staticmethod
    def _last_valid(hidden, mask):
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        index = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
        if (index < 0).any():
            raise ValueError("Cannot pool empty prompt")
        return hidden[torch.arange(len(hidden), device=hidden.device), index]

    def _forward(self, embeds, mask):
        kwargs = {}
        if not self.legacy:
            positions = mask.long().cumsum(-1) - 1
            positions.masked_fill_(mask == 0, 1)
            kwargs["position_ids"] = positions
        out = self.transformer(inputs_embeds=embeds, attention_mask=mask, use_cache=False, return_dict=True, **kwargs)
        hidden = self._last_valid(out.last_hidden_state, mask)
        if hidden.shape[-1] != self.width or not torch.isfinite(hidden).all():
            raise ValueError("Nonfinite/wrong-width Samsone representation")
        return hidden.cpu().clone()

    @torch.no_grad()
    def action(self, choices: list[str]):
        tok = self._tokens(choices)
        embeds = self.model.text_model.get_embeddings_from_tokens(tok["input_ids"])
        return self._forward(embeds, tok["attention_mask"])

    @torch.no_grad()
    def _audio_features(self, path, silence):
        waveform = decode_audio(path, self.helper.sample_rate)
        if len(waveform) > self.helper.max_audio_samples:
            raise ValueError(f"Audio exceeds supported context; refusing label-changing crop: {path}")
        if silence:
            waveform.fill(0)
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        mask = torch.ones_like(waveform)
        waveform, mask = self.helper.crop_or_pad(waveform, mask)
        return self.helper.feature_extractor(waveform, mask)

    @torch.no_grad()
    def _audio_feature_batch(self,audio_paths,silence=False):
        # Many music QA share a clip. Decode/extract it once within the batch,
        # then restore the exact original order before the unchanged encoder.
        unique=list(dict.fromkeys(map(str,audio_paths)))
        values=self._feature_pool.map(lambda path:self._audio_features(path,silence),unique)
        by_path=dict(zip(unique,values))
        features_by_key = {}
        for path in audio_paths:
            features_for_audio=by_path[str(path)]
            for key, value in features_for_audio.items():
                features_by_key.setdefault(key, []).append(value)
        return features_by_key

    @torch.no_grad()
    def state(self, audio_paths: list[str | Path], questions: list[str], *, silence=False):
        if len(audio_paths) != len(questions):
            raise ValueError("Each audio must have a question")
        from samsone.data.constants import AUDIO_TO_SAMPLE_ASSIGNMENT_KEY
        from samsone.data.constants import ATTENTION_MASK_KEY
        features_by_key=self._audio_feature_batch(audio_paths,silence)
        features = {key: torch.cat(values).to(self.device, dtype=(self.compute_dtype
                    if values[0].is_floating_point() and key != ATTENTION_MASK_KEY else values[0].dtype))
                    for key, values in features_by_key.items()}
        audio, audio_mask = self.model.audio_encoder(features)
        projected, audio_mask = self.model.projector(audio, audio_mask)
        pre = self.helper._tokenize("")
        pre = {key: value.expand(len(questions), -1) for key, value in pre.items()}
        post = self._tokens([q + " answer: " for q in questions])
        embeds, mask = self.model._combine_embeds_for_lm_input(
            projected_audio_embeddings=projected, audio_attn_mask=audio_mask,
            metadata_batch={AUDIO_TO_SAMPLE_ASSIGNMENT_KEY: torch.arange(len(questions), device=self.device)},
            pre_audio_text_embeddings=self.model.text_model.get_embeddings_from_tokens(pre["input_ids"]),
            pre_audio_text_attention_mask=pre["attention_mask"], post_audio_tokens_batch=post,
            post_audio_text_embeddings=self.model.text_model.get_embeddings_from_tokens(post["input_ids"]),
            post_audio_text_attention_mask=post["attention_mask"], response_mask_batch=None, max_tokens=None)
        return self._forward(embeds, mask)
