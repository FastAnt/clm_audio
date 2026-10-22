"""Choice-aware bidirectional CLM objective; unknown cross-question pairs are masked."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def canonical_action_ids(choices,device=None):
    # Samsone lowercases text: case variants must not become identical-vector negatives.
    from samsone.inference.models import Samsone356M
    representatives={}
    ids=[]
    for i,text in enumerate(choices):
        normalized=Samsone356M._normalise_prompt(text)
        ids.append(representatives.setdefault(normalized,i))
    return torch.tensor(ids,device=device,dtype=torch.long)


def bidirectional_choice_loss(model, states, actions, candidate_indices, labels, *, mask_unknown=True):
    """One column per exact candidate text ID, with all annotated positives in reverse.

    With mask_unknown=True, an answer belonging only to another row is
    not assumed wrong. The reverse direction uses the transpose of this same
    explicit candidate/label relationship. False reproduces the official CLM
    choice InfoNCE denominator, which assumes other pooled answers are negatives.
    Identical positives never conflict.
    """
    valid = candidate_indices >= 0
    pool, columns = torch.unique(candidate_indices[valid], return_inverse=True)
    mapped = torch.zeros_like(candidate_indices)
    mapped[valid] = columns
    allowed = torch.zeros(len(states), len(pool), dtype=torch.bool, device=states.device)
    row_indices = torch.arange(len(states),device=states.device)[:,None].expand_as(mapped)
    allowed[row_indices[valid],mapped[valid]] = True
    positive_columns = mapped[torch.arange(len(states),device=states.device),labels]
    positive = torch.zeros_like(allowed)
    positive[torch.arange(len(states),device=states.device),positive_columns] = True
    logits = model.scale * model.state(states) @ model.action(actions[pool]).T
    if mask_unknown:
        logits = logits.masked_fill(~allowed, float('-inf'))
    forward = F.cross_entropy(logits,positive_columns)
    keep = positive.any(0)
    reverse_logits = logits.T[keep]
    reverse_positive = positive.T[keep]
    reverse_log_probs = F.log_softmax(reverse_logits,dim=1)
    # Mask explicitly before multiplying: zero * -inf would produce NaNs.
    selected = reverse_log_probs.masked_fill(~reverse_positive,0.)
    reverse = -(selected.sum(1)/reverse_positive.sum(1)).mean()
    return (forward+reverse)/2, dict(forward=forward, reverse=reverse)
