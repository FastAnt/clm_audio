from samsone_clm.precompute import feature_key, random_audio_indices


def test_reuse_checks_provenance_and_shard_integrity(tmp_path, monkeypatch):
    import pytest
    import torch
    from safetensors.torch import save_file
    from samsone_clm import precompute
    from samsone_clm.common import sha256, write_json

    monkeypatch.setattr(precompute, "ROOT", tmp_path)
    directory = tmp_path / "features/spike-fixture"
    directory.mkdir(parents=True)
    rows = [dict(audio_identity="a", question="what?", answer_index=0),
            dict(audio_identity="b", question="when?", answer_index=0)]
    state = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    for mode in ("correct", "random", "silence", "actions"):
        save_file({"hidden": state[:1] if mode == "actions" else state}, str(directory / f"{mode}-000000.safetensors"))
    provenance = dict(hidden_width=3, recipe="fixture")
    write_json(directory / "manifest.json", dict(directory=str(directory), provenance=provenance, rows=rows,
               choices=["a choice"], candidate_indices=[[0], [0]], random_audio_indices=[1, 0],
               shard_sha256={p.name: sha256(p) for p in directory.glob("*.safetensors")}))
    assert precompute.reusable_features(dict(hidden_width=3, recipe="different")) == ({}, [])
    cache, sources = precompute.reusable_features(provenance)
    assert len(sources) == 1
    assert torch.equal(cache[("action", "a choice")], state[0])
    assert torch.equal(cache[("state", "b", "what?")], state[0])
    assert torch.equal(cache[("silence", "b", "when?")], state[1])
    (directory / "correct-000000.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        precompute.reusable_features(provenance)


def test_audio_derangement_handles_multiple_questions_and_preserves_splits():
    rows = [dict(split=split, audio_identity=identity, question=f"question {j}")
            for split in ("train", "validation", "test")
            for identity, count in (("a", 3), ("b", 2), ("c", 1)) for j in range(count)]
    replacements = random_audio_indices(rows)
    assert sorted(replacements) == list(range(len(rows)))
    for i, other in enumerate(replacements):
        assert rows[i]["split"] == rows[other]["split"]
        assert rows[i]["audio_identity"] != rows[other]["audio_identity"]


def test_reuse_key_follows_actual_audio_and_question_not_dataset_row_index():
    rows = [dict(audio_identity="a", question="what?"), dict(audio_identity="b", question="when?")]
    replacements = [1, 0]
    reordered = list(reversed(rows))
    assert feature_key(rows, replacements, "correct", 0) == feature_key(reordered, replacements, "correct", 1)
    assert feature_key(rows, replacements, "random", 0) == ("state", "b", "what?")
    assert feature_key(rows, replacements, "silence", 0) != feature_key(rows, replacements, "correct", 0)


def test_random_music_uses_another_source_parent_not_an_augmentation():
    rows=[dict(split=split,split_group=f'{split}-parent-{parent}',audio_identity=f'{split}-{parent}-{variant}',
               pcm_identity=f'{split}-{parent}-{variant}')
          for split in ('train','validation','test') for parent in range(3) for variant in range(4)]
    replacements=random_audio_indices(rows)
    for i,j in enumerate(replacements):
        assert rows[i]['split']==rows[j]['split']
        assert rows[i]['split_group']!=rows[j]['split_group']
        assert rows[i]['pcm_identity']!=rows[j]['pcm_identity']


def test_precompute_handles_partial_final_shard_and_reuses_completed_store(tmp_path, monkeypatch):
    import json
    import torch
    from samsone_clm import precompute
    from samsone_clm.common import sha256

    calls = []

    class Encoder:
        width = 3
        provenance = {"hidden_width": 3, "recipe": "partial-final-shard-fixture"}

        def __init__(self):
            self.model = torch.nn.Linear(1, 1).requires_grad_(False)

        def action(self, choices):
            calls.append(("action", len(choices)))
            return torch.ones(len(choices), self.width, dtype=torch.bfloat16)

        def state(self, paths, questions, *, silence=False):
            calls.append(("silence" if silence else "state", len(paths)))
            return torch.full((len(paths), self.width), 0 if silence else 1, dtype=torch.bfloat16)

    rows = []
    for split in ("train", "validation", "test"):
        for i in range(3):
            audio = tmp_path/f"{split}-{i}.wav"
            audio.write_bytes(f"source fixture {split} {i}".encode())
            rows.append(dict(audio_path=str(audio), audio_sha256=sha256(audio), audio_identity=f"{split}-{i}",
                             split=split, sample_id=f"{split}-{i}", question="same question",
                             question_type="sound", choices=["yes", "no"], answer_index=0))
    dataset = tmp_path/"dataset.jsonl"
    dataset.write_text("".join(json.dumps(row)+"\n" for row in rows))
    monkeypatch.setattr(precompute, "ROOT", tmp_path)
    monkeypatch.setattr(precompute, "FrozenSamsone", Encoder)
    monkeypatch.setattr(__import__('samsone_clm.common', fromlist=['ROOT']), 'ROOT', tmp_path)
    precompute.precompute(batch_size=3, shard_size=4, dataset=dataset, ablation_splits=["validation", "test"])
    pointer = json.loads((tmp_path/"features/spike_manifest.json").read_text())
    data, manifest = precompute.load_features(pointer["path"], dtype=torch.bfloat16)
    assert len(manifest["rows"]) == 9
    for mode in ("correct", "random", "silence"):
        assert data[mode].shape == (9, 3)
    assert torch.equal(data["correct"], torch.ones(9, 3, dtype=torch.bfloat16))
    assert not data["random"][:3].any()
    assert data["random"][3:].all()
    previous_calls = list(calls)
    precompute.precompute(batch_size=3, shard_size=4, dataset=dataset, ablation_splits=["validation", "test"])
    assert calls == previous_calls
    # Simulate an interrupted run with durable correct/action shards, but no
    # completed manifest. Random audio can reuse those states without rerunning
    # the frozen encoder, because its prompt is unchanged.
    directory=__import__('pathlib').Path(precompute.resolve_data_path(pointer['path'])).parent
    (directory/'manifest.json').unlink()
    for mode in ('random','silence'):
        for path in directory.glob(mode+'-*.safetensors'):path.unlink()
    precompute.precompute(batch_size=3,shard_size=4,dataset=dataset,ablation_splits=['validation','test'])
    assert all(mode=='silence' for mode,count in calls[len(previous_calls):])
    data,manifest=precompute.load_features(pointer['path'],dtype=torch.bfloat16)
    assert manifest['extraction_counts']['correct']['resumed']==9
    assert manifest['extraction_counts']['random']['computed']==0
