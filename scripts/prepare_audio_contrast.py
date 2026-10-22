"""Freeze the 10K pilot plus verified auxiliary TRAIN rows, without changing held-out rows."""
from collections import Counter
import argparse
import json

from samsone_clm.common import ROOT, read_rows, sha256, write_json
from samsone_clm.contrast import supervised_audio_partners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-mode", choices=("same-options", "reciprocal-answers"), default="same-options")
    parser.add_argument("--base-snapshot", choices=("stage1_10k", "stage2_50k", "stage3_150k", "stage4_all"), default="stage1_10k")
    parser.add_argument("--data-dir",type=__import__('pathlib').Path,default=ROOT/'processed/v2')
    args = parser.parse_args()
    suffix = "" if args.pair_mode == "same-options" else "_reciprocal"
    directory = args.data_dir
    canonical = directory / "canonical.jsonl"
    base_path = directory / f"{args.base_snapshot}.jsonl"
    full = read_rows(canonical)
    train = [i for i, row in enumerate(full) if row["split"] == "train"]
    partners = supervised_audio_partners(full, train, args.pair_mode)
    pair_rows = (partners >= 0).nonzero().flatten().tolist()
    base = read_rows(base_path)
    base_ids = {row["sample_id"] for row in base}
    extra = [dict(full[i], training_role="contrast_only") for i in pair_rows
             if full[i]["sample_id"] not in base_ids]
    rows = base + extra
    heldout = [row for row in full if row["split"] != "train"]
    train_rows = [row for row in rows if row["split"] == "train"]
    source_overlap = {r["audio_identity"] for r in train_rows} & {r["audio_identity"] for r in heldout}
    pcm_overlap = {r.get("pcm_identity", r["audio_identity"]) for r in train_rows} & {
        r.get("pcm_identity", r["audio_identity"]) for r in heldout}
    assert not source_overlap and not pcm_overlap
    assert len({r["sample_id"] for r in rows}) == len(rows)
    assert all(full[i]["split"] == full[int(partners[i])]["split"] == "train" for i in pair_rows)
    assert [r for r in rows if r["split"] != "train"] == [r for r in base if r["split"] != "train"]
    output_name = "stage2_audio_contrast" if args.base_snapshot == "stage1_10k" else args.base_snapshot + "_audio_contrast"
    report_suffix = suffix if args.base_snapshot == "stage1_10k" else "_" + args.base_snapshot + suffix
    path = directory / f"{output_name}{suffix}.jsonl"
    contents = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if path.exists() and path.read_text() != contents:
        raise ValueError("Existing immutable contrast snapshot differs")
    path.write_text(contents)
    pairs_path = directory / f"audio_contrast_pairs{report_suffix}.json"
    write_json(pairs_path, [dict(sample_id=full[i]["sample_id"],
                               partner_sample_id=full[int(partners[i])]["sample_id"])
                           for i in pair_rows])
    report = dict(dataset=str(path), dataset_sha256=sha256(path),
                  canonical_sha256=sha256(canonical), base_dataset_sha256=sha256(base_path),
                  main_train_samples=sum(r["split"] == "train" for r in base),
                  auxiliary_train_samples=len(extra), total_train_samples=len(train_rows),
                  directed_paired_rows=len(pair_rows),
                  paired_types=dict(Counter(full[i]["question_type"] for i in pair_rows)),
                  validation_samples=sum(r["split"] == "validation" for r in rows),
                  test_samples=sum(r["split"] == "test" for r in rows),
                  heldout_unchanged=True, heldout_audio_overlap=len(source_overlap),
                  heldout_pcm_overlap=len(pcm_overlap), pair_manifest=str(pairs_path), pair_mode=args.pair_mode)
    write_json(directory / f"audio_contrast{report_suffix}.report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
