"""Freeze the complete corpus, nested train prefixes and fixed stratified held-out samples."""
import json
import shutil
from collections import Counter

from samsone_clm.common import ROOT, digest, read_rows, sha256, write_json


def round_robin(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["question_type"], []).append(row)
    for values in groups.values():
        values.sort(key=lambda row: digest([42, row["sample_id"]]))
    ordered = []
    for i in range(max(map(len, groups.values()))):
        for category in sorted(groups):
            if i < len(groups[category]):
                ordered.append(groups[category][i])
    return ordered


def main():
    directory = ROOT / "processed/v2"
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / "processed/full.jsonl"
    frozen = directory / "canonical.jsonl"
    if not frozen.exists():
        temporary = frozen.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(frozen)
    if sha256(source) != sha256(frozen):
        raise ValueError("Canonical corpus changed after v2 freeze")
    rows = read_rows(frozen)
    train = round_robin([row for row in rows if row["split"] == "train"])
    validation = round_robin([row for row in rows if row["split"] == "validation"])[:2048]
    test = round_robin([row for row in rows if row["split"] == "test"])[:2048]
    snapshots = {}
    for name, count, heldout in (("pilot_1000", 1000, 256), ("stage1_10k", 10000, 2048),
                                 ("stage2_50k", 50000, 2048), ("stage3_150k", 150000, 2048),
                                 ("stage4_all", len(train), 2048)):
        selection = train[:count] + validation[:heldout] + test[:heldout]
        path = directory / f"{name}.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selection)
        if path.exists() and path.read_text() != text:
            raise ValueError(f"Refusing to change fixed snapshot: {path}")
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(text)
            temporary.replace(path)
        snapshots[name] = dict(path=str(path), sha256=sha256(path), train=count,
                                validation=heldout, test=heldout,
                                train_types=dict(Counter(row["question_type"] for row in train[:count])))
    report = dict(canonical_sha256=sha256(frozen), snapshots=snapshots,
                  selected_samples={s:sum(row["split"]==s for row in rows) for s in ("train","validation","test")},
                  stage_validation_ids=[row["sample_id"] for row in validation],
                  stage_test_ids=[row["sample_id"] for row in test], nested_train=True,
                  sampling="category round-robin of immutable hash orders; train prefixes; unchanged identity splits")
    write_json(directory / "snapshots.json", report)
    write_json(directory / "canonical.report.json", report)
    print(json.dumps({name:{k:v for k,v in value.items() if k not in ("path","sha256")} for name,value in snapshots.items()},indent=2))


if __name__ == "__main__":
    main()
