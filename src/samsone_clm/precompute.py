from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict, deque

import torch
from safetensors.torch import load_file, save_file

from .common import ROOT, ORIGINAL_DATA_ROOT, digest, read_rows, sha256, write_json, resolve_data_path
from .encoder import FrozenSamsone


def random_audio_indices(rows):
    replacements = list(range(len(rows)))
    for split in ("train", "validation", "test"):
        group = lambda i: rows[i].get('split_group',rows[i]['audio_identity'])
        positions = sorted((i for i, row in enumerate(rows) if row["split"] == split),key=group)
        counts = Counter(group(i) for i in positions)
        shift = max(counts.values(), default=0)
        if len(positions) < 2 or shift * 2 > len(positions):
            raise ValueError(f"Cannot derange audio identities in {split}")
        def valid_offset(offset):
            return all(group(i) != group(positions[(j+offset)%len(positions)]) and
                       rows[i].get('pcm_identity',rows[i]['audio_identity']) !=
                       rows[positions[(j+offset)%len(positions)]].get('pcm_identity',rows[positions[(j+offset)%len(positions)]]['audio_identity'])
                       for j,i in enumerate(positions))
        if not valid_offset(shift):
            import random
            offsets = range(shift,len(positions)-shift+1)
            candidates = random.Random(42).sample(offsets,min(64,len(offsets)))
            shift = next((offset for offset in candidates if valid_offset(offset)),None)
            if shift is None:
                raise ValueError(f'Cannot derange source groups and PCM in {split}')
        for j,i in enumerate(positions):
            replacements[i] = positions[(j+shift)%len(positions)]
    return replacements



def select_ablation_positions(rows, splits, limit=None):
    """Keep every question form of a selected genre recording together.

    Round-robin semantic classes prevents common genres from hiding rare ones.
    Existing non-genre cohorts retain their exact historical hash ordering.
    """
    selected = set()
    for split in splits:
        regular = defaultdict(list)
        genre = defaultdict(lambda: defaultdict(list))
        for i, row in enumerate(rows):
            if row["split"] != split:
                continue
            if row.get("genre_corpus"):
                label = row.get("semantic_label", row["choices"][row["answer_index"]])
                parent = row.get("parent_sample_id", row["audio_identity"])
                genre[(row["question_type"], row.get("music_task", ""), label)][parent].append(i)
            else:
                regular[(row["question_type"], row.get("music_task", ""))].append(i)
        groups = {}
        for key, values in regular.items():
            ordered = sorted(values, key=lambda i: digest([42, rows[i]["sample_id"]]), reverse=True)
            groups[("regular", *key)] = deque([i] for i in ordered)
        for key, parents in genre.items():
            ordered = sorted(parents, key=lambda parent: digest(["genre-evaluation-parent-v1", parent]))
            groups[("genre", *key)] = deque(parents[parent] for parent in ordered)
        positions = []
        while any(groups.values()):
            progressed = False
            for key in sorted(groups):
                if not groups[key]:
                    continue
                block = groups[key][0]
                if limit is not None and len(positions) + len(block) > limit:
                    continue
                positions.extend(groups[key].popleft())
                progressed = True
            if not progressed:
                break
        selected.update(positions)
    return selected


def feature_key(rows, replacements, mode, index):
    source = replacements[index] if mode == "random" else index
    if mode == 'silence' and 'audio_num_samples' in rows[index]:
        # Exact waveform length controls the native attention mask. Music rows
        # record it explicitly, so equal-duration silence/questions are identical.
        return ('silence', f"samples:{rows[index]['audio_num_samples']}", rows[index]['question'])
    return ("silence" if mode == "silence" else "state", rows[source].get("pcm_identity", rows[source]["audio_identity"]), rows[index]["question"])


def reusable_features(encoder_provenance):
    cache, sources = {}, []
    for path in sorted((ROOT / "features").glob("spike-*/manifest.json")):
        old = json.loads(path.read_text())
        if any(old["provenance"].get(k) != v for k, v in encoder_provenance.items()):
            continue
        data, old = load_features(str(path), dtype=torch.bfloat16)  # Validate hashes before reuse.
        for text, hidden in zip(old["choices"], data["actions"]):
            cache[("action", text)] = hidden.to(torch.bfloat16)
        for mode in ("correct", "random", "silence"):
            for i, hidden in enumerate(data[mode]):
                if mode != "correct" and not data["ablation_available"][i]:
                    continue
                cache[feature_key(old["rows"], old["random_audio_indices"], mode, i)] = hidden.to(torch.bfloat16)
        sources.append(str(path))
    return cache, sources


def precompute(batch_size=8, shard_size=64, dataset=None, ablation_splits=None, ablation_limit=None):
    torch.set_num_threads(2)
    rows = read_rows(dataset or (ROOT/"processed/full.jsonl" if (ROOT/"processed/full.jsonl").exists() else ROOT/"processed/spike.jsonl"))
    ablation_splits = sorted(ablation_splits or ["train", "validation", "test"])
    ablation_positions = select_ablation_positions(rows, ablation_splits, ablation_limit)
    checked_paths = set()
    expected_audio = {r["audio_path"]:r["audio_sha256"] for r in rows}
    def verify_paths(paths):
        for path in paths:
            if path not in checked_paths:
                if sha256(__import__("pathlib").Path(path)) != expected_audio[path]:
                    raise ValueError(f"Audio changed after prepare: {path}")
                checked_paths.add(path)
    encoder = FrozenSamsone()
    choices = sorted({text for row in rows for text in row["choices"]})
    ids = {text: i for i, text in enumerate(choices)}
    # WAV FLOAT headers can contain timestamps: cache identity uses decoded PCM identity, not container headers.
    semantic_rows = [{k: v for k, v in row.items() if k != "audio_sha256"} for row in rows]
    provenance = {**encoder.provenance, "dataset_semantic_sha256": digest(semantic_rows),
                  "choices_sha256": digest(choices), "random_recipe": ("same-split source-group and PCM derangement v3"
                   if any('split_group' in row for row in rows) else "same-split identity-group derangement v2"),
                  "ablation_splits": ablation_splits}
    if ablation_limit is not None:
        provenance["ablation_limit_per_split"] = ablation_limit
    if any(row.get("genre_corpus") for row in rows):
        provenance["ablation_recipe"] = "semantic-class round-robin, complete recording/question groups v1"
    directory = ROOT / "features" / ("spike-" + digest(provenance)[:16])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        existing = json.loads((directory / "manifest.json").read_text())
        if existing["provenance"] != provenance:
            raise ValueError("Feature cache provenance collision")
        load_features(str(directory / "manifest.json"))
        write_json(ROOT / "features/spike_manifest.json", {"path": str(ORIGINAL_DATA_ROOT/directory.relative_to(ROOT)/"manifest.json"),
                   "sha256": sha256(directory / "manifest.json")})
        print(f"Reusing all existing frozen features: {directory}", flush=True)
        return
    write_json(directory / "provenance.json", provenance)
    cache, cache_sources = reusable_features(encoder.provenance)
    extraction_counts = {mode: {"reused": 0, "computed": 0, "resumed": 0, "not_requested": 0} for mode in ("actions", "correct", "random", "silence")}
    print(f"Reusable representations: {len(cache)} from {len(cache_sources)} validated feature stores", flush=True)
    start = time.monotonic()
    for first in range(0, len(choices), shard_size):
        path = directory / f"actions-{first:06d}.safetensors"
        subset = choices[first:min(first + shard_size, len(choices))]
        if path.exists():
            values=load_file(str(path))['hidden']
            if values.shape!=(len(subset),encoder.width) or not torch.isfinite(values).all():
                raise ValueError(f'Invalid resumed action shard: {path}')
            cache.update((('action',text),hidden) for text,hidden in zip(subset,values))
            extraction_counts['actions']['resumed']+=len(subset)
            continue
        missing = [text for text in subset if ("action", text) not in cache]
        for i in range(0, len(missing), batch_size * 4):
            texts = missing[i:i + batch_size * 4]
            for text, hidden in zip(texts, encoder.action(texts)):
                cache[("action", text)] = hidden
        extraction_counts["actions"]["computed"] += len(missing)
        extraction_counts["actions"]["reused"] += len(subset) - len(missing)
        save_file({"hidden": torch.stack([cache[("action", text)] for text in subset]).to(torch.bfloat16)}, str(path.with_suffix(".tmp")))
        path.with_suffix(".tmp").replace(path)
        print(f"action hidden {min(first + shard_size, len(choices))}/{len(choices)}", flush=True)
    replacements = random_audio_indices(rows)
    for mode in ("correct", "random", "silence"):
        for first in range(0, len(rows), shard_size):
            path = directory / f"{mode}-{first:06d}.safetensors"
            if path.exists():
                positions=list(range(first,min(first+shard_size,len(rows))))
                values=load_file(str(path))['hidden']
                if values.shape!=(len(positions),encoder.width) or not torch.isfinite(values).all():
                    raise ValueError(f'Invalid resumed state shard: {path}')
                requested=[i for i in positions if mode=='correct' or i in ablation_positions]
                for i,hidden in zip(positions,values):
                    if mode=='correct' or i in ablation_positions:
                        cache[feature_key(rows,replacements,mode,i)]=hidden
                extraction_counts[mode]['resumed']+=len(requested)
                extraction_counts[mode]['not_requested']+=len(positions)-len(requested)
                continue
            hidden = []
            for offset in range(first, min(first + shard_size, len(rows)), batch_size):
                positions = list(range(offset, min(offset + batch_size, first + shard_size, len(rows))))
                requested = [i for i in positions if mode == "correct" or i in ablation_positions]
                if mode == "correct":
                    verify_paths([rows[i]["audio_path"] for i in requested])
                keys = [feature_key(rows, replacements, mode, i) for i in requested]
                missing = list(dict.fromkeys(key for key in keys if key not in cache))
                key_to_index = dict(zip(keys, requested))
                if missing:
                    pending = [key_to_index[key] for key in missing]
                    paths = [rows[replacements[i] if mode == "random" else i]["audio_path"] for i in pending]
                    verify_paths(paths)
                    values = encoder.state(paths, [rows[i]["question"] for i in pending], silence=mode == "silence")
                    cache.update(zip(missing, values))
                extraction_counts[mode]["computed"] += len(missing)
                extraction_counts[mode]["reused"] += len(keys) - len(missing)
                extraction_counts[mode]["not_requested"] += len(positions) - len(requested)
                requested_set = set(requested)
                hidden.append(torch.stack([cache[feature_key(rows, replacements, mode, i)] if i in requested_set
                                           else torch.zeros(encoder.width, dtype=torch.bfloat16) for i in positions]))
            save_file({"hidden": torch.cat(hidden).to(torch.bfloat16)}, str(path.with_suffix(".tmp")))
            path.with_suffix(".tmp").replace(path)
            print(f"{mode} hidden {min(first + shard_size, len(rows))}/{len(rows)}, elapsed={time.monotonic()-start:.1f}s", flush=True)
    logical_directory = ORIGINAL_DATA_ROOT/directory.relative_to(ROOT)
    manifest = dict(provenance=provenance, directory=str(logical_directory), rows=rows, choices=choices,
                    candidate_indices=[[ids[c] for c in row["choices"]] for row in rows], shard_size=shard_size,
                    random_audio_indices=replacements, seconds=time.monotonic()-start,
                    extraction_counts=extraction_counts, reused_feature_manifests=cache_sources,
                    ablation_available=[i in ablation_positions for i in range(len(rows))],
                    shard_sha256={p.name: sha256(p) for p in directory.glob("*.safetensors")})
    write_json(directory / "manifest.json", manifest)
    write_json(ROOT / "features/spike_manifest.json", {"path": str(logical_directory / "manifest.json"),
               "sha256": sha256(directory / "manifest.json")})
    # Ensure no model weights changed and that the frozen paths actually contain audio/question information.
    states = torch.cat([load_file(str(p))["hidden"] for p in sorted(directory.glob("correct-*.safetensors"))])
    silence = torch.cat([load_file(str(p))["hidden"] for p in sorted(directory.glob("silence-*.safetensors"))])
    audit = dict(frozen_parameters=all(not p.requires_grad for p in encoder.model.parameters()),
                 total_parameters=sum(p.numel() for p in encoder.model.parameters()),
                 state_shape=list(states.shape), audio_changed_mean_l2=(states[manifest["ablation_available"]].float()-silence[manifest["ablation_available"]].float()).norm(dim=-1).mean().item(),
                 nonfinite=int((~torch.isfinite(states)).sum()), provenance=provenance)
    assert audit["frozen_parameters"] and not audit["nonfinite"] and audit["audio_changed_mean_l2"] > 0
    write_json(directory / "extraction_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


def load_features(path=None, dtype=torch.float32):
    path = path or json.loads((ROOT / "features/spike_manifest.json").read_text())["path"]
    manifest = json.loads(resolve_data_path(path).read_text())
    directory = resolve_data_path(manifest["directory"])
    for name, expected in manifest["shard_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Feature shard checksum mismatch: {name}")
    data = {mode: torch.cat([load_file(str(p))["hidden"] for p in sorted(directory.glob(f"{mode}-*.safetensors"))]).to(dtype)
            for mode in ("correct", "random", "silence", "actions")}
    candidates = manifest["candidate_indices"]
    if not candidates or any(not values for values in candidates):
        raise ValueError("Every row requires candidates")
    maximum = max(map(len, candidates))
    # Padding repeats a valid cached action; the mask excludes it from all scores.
    # Do not mutate the immutable, hashed manifest to store padding.
    data["indices"] = torch.tensor([values + [values[0]] * (maximum-len(values)) for values in candidates], dtype=torch.long)
    data["candidate_mask"] = torch.tensor([[True]*len(values)+[False]*(maximum-len(values)) for values in candidates])
    data["labels"] = torch.tensor([r["answer_index"] for r in manifest["rows"]])
    if any(not 0 <= row['answer_index'] < len(values) for row, values in zip(manifest['rows'], candidates)):
        raise ValueError("Correct answer outside candidate list")
    data["ablation_available"] = torch.tensor(manifest.get("ablation_available", [True] * len(manifest["rows"])), dtype=torch.bool)
    for name in ("correct", "random", "silence"):
        if len(data[name]) != len(manifest["rows"]):
            raise ValueError(f"Incomplete {name} feature store")
    if len(data["actions"]) != len(manifest["choices"]):
        raise ValueError("Incomplete action feature store")
    for name in ("correct", "random", "silence", "actions"):
        if data[name].shape[-1] != manifest["provenance"]["hidden_width"] or not torch.isfinite(data[name]).all():
            raise ValueError(f"Invalid representation in {name}")
    return data, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=64)
    ap.add_argument("--dataset")
    ap.add_argument('--pointer', help='Save a named output pointer for this dataset component')
    ap.add_argument("--ablation-splits", nargs="+", choices=["train", "validation", "test"])
    ap.add_argument("--ablation-limit-per-split", type=int, help="Deterministic stratified ablation subset; correct-audio evaluation still covers the full split")
    args = ap.parse_args()
    precompute(args.batch_size, args.shard_size, args.dataset, args.ablation_splits, args.ablation_limit_per_split)
    if args.pointer:
        pointer = __import__('pathlib').Path(args.pointer)
        write_json(pointer,json.loads((ROOT/'features/spike_manifest.json').read_text()))


if __name__ == "__main__":
    main()
