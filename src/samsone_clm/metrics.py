from __future__ import annotations

import torch
from torch.nn import functional as F


def metrics(scores, labels, bins=15):
    probs = scores.softmax(-1)
    confidence, predicted = probs.max(-1)
    correct = predicted.eq(labels)
    order = scores.topk(min(2, scores.shape[-1]), dim=-1).indices
    ece = 0.0
    for b in range(bins):
        mask = (confidence > b / bins) & (confidence <= (b + 1) / bins)
        if mask.any():
            ece += float(mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs())
    return dict(samples=len(labels), top1=correct.float().mean().item(),
                top2=order.eq(labels[:, None]).any(-1).float().mean().item(),
                loss=F.cross_entropy(scores, labels).item(),
                mean_correct_probability=probs[torch.arange(len(labels), device=labels.device), labels].mean().item(),
                ece=ece, bins=bins)


def music_task_metrics(scores, labels, rows):
    result = metrics(scores, labels)
    # Classes are answer TEXTS, not shuffled choice column numbers.
    answers = [row.get('semantic_label',row['choices'][row['answer_index']]) for row in rows]
    correct = scores.argmax(-1).eq(labels)
    values = []
    counts = {}
    for answer in sorted(set(answers)):
        mask = torch.tensor([text == answer for text in answers],device=scores.device)
        values.append(correct[mask].float().mean().item())
        counts[answer] = int(mask.sum())
    independent=len({row.get('parent_sample_id',row.get('sample_id',('fixture',i))) for i,row in enumerate(rows)})
    result.update(class_macro_top1=sum(values)/len(values), semantic_class_counts=counts,independent_qa_samples=independent)
    return result


@torch.no_grad()
def evaluate_model(model, data, rows, indices, batch_size=64, *, include_views=True, include_genre=True):
    result = {}
    for mode in ("correct", "random", "silence"):
        available = data.get("ablation_available")
        mode_indices = indices if mode == "correct" or available is None else indices[available[indices]]
        if not len(mode_indices):
            result[mode] = {"overall": {"samples": 0}, "reason": "Audio ablation was not precomputed for this split"}
            continue
        predictions = []
        for chunk in mode_indices.split(batch_size):
            mask = data.get('candidate_mask')
            scores = model(data[mode][chunk], data["actions"][data["indices"][chunk]])
            predictions.append(scores.masked_fill(~mask[chunk], float('-inf')) if mask is not None else scores)
        scores = torch.cat(predictions)
        labels = data["labels"][mode_indices]
        mode_result = {"overall": metrics(scores, labels)}
        for category in ("music", "sound", "speech", "temporal"):
            mask = torch.tensor([rows[i]["question_type"].lower() == category for i in mode_indices.tolist()], device=scores.device)
            mode_result[category] = metrics(scores[mask], labels[mask]) if mask.any() else {"samples": 0}
        result[mode] = mode_result
        tasks = sorted({rows[i].get('music_task') for i in mode_indices.tolist() if rows[i].get('music_task')})
        if tasks:
            mode_result['music_tasks'] = {}
            for task in tasks:
                mask = torch.tensor([rows[i].get('music_task') == task for i in mode_indices.tolist()], device=scores.device)
                selected_rows = [rows[i] for i in mode_indices.tolist() if rows[i].get('music_task') == task]
                mode_result['music_tasks'][task] = music_task_metrics(scores[mask], labels[mask], selected_rows)
        if mode == "correct" and available is not None and available[indices].any():
            subset = available[indices]
            reference = {"overall": metrics(scores[subset], labels[subset])}
            for category in ("music", "sound", "speech", "temporal"):
                mask = torch.tensor([rows[i]["question_type"].lower() == category for i in indices.tolist()], device=scores.device) & subset
                reference[category] = metrics(scores[mask], labels[mask]) if mask.any() else {"samples": 0}
            result["ablation_correct"] = reference
            if tasks:
                reference['music_tasks'] = {}
                for task in tasks:
                    mask = torch.tensor([rows[i].get('music_task') == task for i in indices.tolist()], device=scores.device) & subset
                    if mask.any():
                        selected_rows = [rows[i] for i in indices[mask].tolist()]
                        reference['music_tasks'][task] = music_task_metrics(scores[mask], labels[mask], selected_rows)
    reference = result.get("ablation_correct", result["correct"])
    result["audio_top1_gaps"] = {mode: reference["overall"]["top1"] - result[mode]["overall"]["top1"]
                                  for mode in ("random", "silence") if "top1" in result[mode]["overall"] and result[mode]["overall"]["samples"] == reference["overall"]["samples"]}
    tasks = result['correct'].get('music_tasks', {})
    if tasks:
        selection_tasks = {key:value for key,value in tasks.items() if value.get('independent_qa_samples',value['samples']) >= 50}
        if not selection_tasks:
            selection_tasks = tasks  # Small diagnostic fixtures; full runs report their sample counts.
        result['music_macro'] = {key: sum(value[key] for value in selection_tasks.values())/len(selection_tasks)
                                 for key in ('top1', 'loss', 'ece')}
        result['music_macro']['top1'] = sum(value['class_macro_top1'] for value in selection_tasks.values())/len(selection_tasks)
        result['music_macro']['selection_tasks'] = list(selection_tasks)
        family = lambda task: 'structure' if task in ('tempo','meter','key_tonic','key_mode','chord','harmony') else 'perception'
        families = sorted({family(task) for task in selection_tasks})
        family_scores = {name:sum(value['class_macro_top1'] for task,value in selection_tasks.items() if family(task)==name)/
                        sum(family(task)==name for task in selection_tasks) for name in families}
        result['music_macro']['family_top1'] = family_scores
        result['music_macro']['top1'] = sum(family_scores.values())/len(family_scores)
        gaps = {}
        family_gaps = {name:{} for name in families}
        for mode in ('random', 'silence'):
            values = []
            for task, own in reference.get('music_tasks', {}).items():
                other = result[mode].get('music_tasks', {}).get(task)
                if task in selection_tasks and other and own['samples'] == other['samples']:
                    values.append(own['class_macro_top1']-other['class_macro_top1'])
            if values:
                gaps[mode] = sum(values)/len(values)
            for name in families:
                differences = []
                for task,own in reference.get('music_tasks',{}).items():
                    other = result[mode].get('music_tasks',{}).get(task)
                    if task in selection_tasks and family(task)==name and other and own['samples']==other['samples']:
                        differences.append(own['class_macro_top1']-other['class_macro_top1'])
                if differences:
                    family_gaps[name][mode] = sum(differences)/len(differences)
            if all(mode in values for values in family_gaps.values()):
                gaps[mode] = sum(values[mode] for values in family_gaps.values())/len(family_gaps)
        result['music_macro']['audio_top1_gaps'] = gaps
        result['music_macro']['family_audio_top1_gaps'] = family_gaps
    views={rows[i].get('music_view','full') for i in indices.tolist()}
    if include_views and 'serving-window' in views:
        result['music_views']={}
        for view in sorted(views):
            selected=torch.tensor([rows[i].get('music_view','full')==view for i in indices.tolist()],device=indices.device)
            result['music_views'][view]=evaluate_model(model,data,rows,indices[selected],batch_size,include_views=False)
    if include_genre and any(rows[i].get('genre_corpus') for i in indices.tolist()):
        genre=torch.tensor([bool(rows[i].get('genre_corpus')) for i in indices.tolist()],device=indices.device)
        result['genre_evaluation']=evaluate_model(model,data,rows,indices[genre],batch_size,include_views=False,include_genre=False)
        result['genre_question_variants']={}
        for case in sorted({rows[i].get('question_variant_id','default') for i in indices[genre].tolist()}):
            selected=torch.tensor([rows[i].get('question_variant_id','default')==case for i in indices[genre].tolist()],device=indices.device)
            result['genre_question_variants'][case]=evaluate_model(model,data,rows,indices[genre][selected],batch_size,include_views=False,include_genre=False)
        if (~genre).any():
            result['replay_evaluation']=evaluate_model(model,data,rows,indices[~genre],batch_size,include_views=False,include_genre=False)
    return result


def genre_selection_key(evaluation, initial):
    genre=evaluation['genre_evaluation']['music_macro']
    cases=evaluation['genre_question_variants']
    gap=min(g for report in cases.values() for g in report['music_macro']['audio_top1_gaps'].values())
    retained=True
    if 'replay_evaluation' in initial:
        before=initial['replay_evaluation']['music_macro']['family_top1']
        after=evaluation['replay_evaluation']['music_macro']['family_top1']
        retained=all(after[name]>=value-.01 for name,value in before.items())
    worst=min(report['music_macro']['top1'] for report in cases.values())
    before=initial['genre_evaluation']['music_macro']['top1']
    before_worst=min(report['music_macro']['top1'] for report in initial['genre_question_variants'].values())
    improved=genre['top1']>=before+.02 and worst>=before_worst+.01
    return (int(gap>=.05 and retained and improved),worst,genre['top1'],gap,-genre['loss'])


def music_selection_key(evaluation, initial=None):
    """Audio gates; optional serving-window improvement with full-clip retention."""
    music=evaluation['music_macro']
    gap=min((gap for values in music['family_audio_top1_gaps'].values() for gap in values.values()),default=float('-inf'))
    if initial is None:
        return (int(gap>=.05),music['top1'],gap,-music['loss'])
    views=evaluation['music_views'];before=initial['music_views']
    full=views['full']['music_macro']['top1'];window=views['serving-window']['music_macro']['top1']
    gaps=[gap]+[g for report in views.values() for values in report['music_macro']['family_audio_top1_gaps'].values() for g in values.values()]
    gap=min(gaps)
    baseline=before['full']['music_macro']
    retained=full>=baseline['top1']-.01 and all(
        views['full']['music_macro']['family_top1'][family]>=score-.01
        for family,score in baseline['family_top1'].items())
    improved=window>=before['serving-window']['music_macro']['top1']+.01
    return (int(gap>=.05 and retained and improved),music['top1'],window,gap,-music['loss'])
