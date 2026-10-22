"""Label-verified audio negatives: identical question/options, different labeled answer."""
from collections import defaultdict

import torch
from torch.nn import functional as F

from .common import digest


def contrast_key(row):
    return (row["question"].lower().strip(),
            tuple(sorted(choice.lower().strip() for choice in row["choices"])))


def supervised_swap_loss(model, data, paired, own_scores, margin):
    """Replace audio while keeping each original row's candidate order and label."""
    # `paired` indexes the original rows. Only state features use partner rows;
    # looking up actions through partner rows would compare different columns.
    swapped_scores = model(data["correct"][data["partners"][paired]],
                           data["actions"][data["indices"][paired]])
    labels = data["labels"][paired, None]
    gap = (own_scores.gather(1, labels) - swapped_scores.gather(1, labels)) / model.scale
    return F.relu(margin - gap).mean()


def sample_audio_pair_batch(partners, paired_rows, pair_count):
    """Sample verified TRAIN pairs, including both endpoints for positive CE."""
    if pair_count <= 0 or not len(paired_rows):
        raise ValueError("Dedicated audio batches require a positive count and verified pairs")
    anchors = paired_rows[torch.randint(len(paired_rows), (pair_count,), device=paired_rows.device)]
    return torch.cat((anchors, partners[anchors]))


def reciprocal_answer_partners(rows, training_indices, diverse=False):
    """Both correct answers must occur in BOTH option sets for the same question.

    The labels then verify that A is wrong for B's audio and B is wrong for A's
    audio, even when the remaining distractors differ. No guessed random negatives.
    """
    buckets = defaultdict(list)
    valid = []
    for i in training_indices:
        row = rows[i]
        choices = [c.lower().strip() for c in row["choices"]]
        if len(set(choices)) != len(choices):
            continue
        question = row["question"].lower().strip()
        answer = choices[row["answer_index"]]
        valid.append((i, question, answer, set(choices)-{answer}))
        for other in set(choices)-{answer}:
            buckets[question, answer, other].append(i)
    partners = torch.full((len(rows),), -1, dtype=torch.long)
    for i, question, answer, alternatives in valid:
        row = rows[i]
        # Preserve the historical recipe unless explicitly requested. Music
        # has fixed prompts/options across thousands of recordings: first-only
        # lookup would reuse a handful of negative recordings at every update.
        seed = int(digest(['audio-pair-diversity-v1',row.get('sample_id',row['audio_identity'])])[:16],16)
        alternatives = sorted(alternatives)
        if diverse and alternatives:
            rotation=seed % len(alternatives)
            alternatives=alternatives[rotation:]+alternatives[:rotation]
        for alternative in alternatives:
            candidates=buckets.get((question, alternative, answer), [])
            start=(seed//max(1,len(alternatives))) % len(candidates) if diverse and candidates else 0
            for offset in range(len(candidates)):
                j=candidates[(start+offset) % len(candidates)]
                other = rows[j]
                if (other["audio_identity"] != row["audio_identity"] and
                    other.get("pcm_identity", other["audio_identity"]) != row.get("pcm_identity", row["audio_identity"]) and
                    (not diverse or other.get('split_group',other['audio_identity']) != row.get('split_group',row['audio_identity']))):
                    partners[i] = j
                    break
            if partners[i] >= 0:
                break
    return partners


def supervised_audio_partners(rows, training_indices, mode="same-options"):
    training_indices = list(training_indices)
    if mode in ("reciprocal-answers", "reciprocal-diverse"):
        return reciprocal_answer_partners(rows, training_indices, diverse=mode=="reciprocal-diverse")
    if mode != "same-options":
        raise ValueError(f"Unknown audio-pair mode: {mode}")
    groups = defaultdict(list)
    for i in training_indices:
        row = rows[i]
        key = contrast_key(row)
        groups[key].append(i)
    partners = torch.full((len(rows),), -1, dtype=torch.long)
    for indices in groups.values():
        for i in indices:
            row = rows[i]
            answer = row["choices"][row["answer_index"]].lower().strip()
            for j in indices:
                other = rows[j]
                if (other["audio_identity"] != row["audio_identity"] and
                    other.get("pcm_identity", other["audio_identity"]) != row.get("pcm_identity", row["audio_identity"]) and
                    other["choices"][other["answer_index"]].lower().strip() != answer):
                    partners[i] = j
                    break
    return partners
