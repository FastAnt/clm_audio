"""Diagnostic-only native Samsone generation and exact adapter/hidden-path audit.

Generation is confined to this baseline audit, never used by training or ranking API.
"""
import argparse
import json
import shutil
import time

import soundfile as sf
import torch
from torch.nn import functional as F

from samsone_clm.common import PROJECT, ROOT, write_json
from samsone_clm.encoder import FrozenSamsone
from samsone_clm.model import AudioCLM


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--max-new-tokens",type=int,default=64)
    args=parser.parse_args()
    torch.set_num_threads(8 if args.device=="cpu" else 2)
    directory=ROOT/f"runs/samsone_native_audit_{args.device}"
    directory.mkdir(parents=True,exist_ok=True)
    audio_dir=ROOT/"cache/samsone_native_audit_audio"
    audio_dir.mkdir(parents=True,exist_ok=True)
    cases=[]
    for name,expected in (("cow","cow mooing / animal vocalizations"),("water","water / rain / splashing")):
        target=audio_dir/f"{name}.wav"
        shutil.copyfile(PROJECT/f"vendor/samsone/assets/{name}.wav",target)
        cases.append(dict(name=f"official_asset_{name}",audio_path=str(target),question="describe",
                          expected_source_description=expected,source="official Samsone repository asset"))
    selected=set()
    with (ROOT/"processed/full.jsonl").open() as source:
        for line in source:
            row=json.loads(line)
            kind=row["question_type"]
            if kind not in ("music","sound","speech") or kind in selected:
                continue
            answer=row["choices"][row["answer_index"]]
            if kind=="music" and not any(word in answer.lower() for word in ("guitar","piano")):
                continue
            if kind=="sound" and not any(word in answer.lower() for word in ("barking","rain","water")):
                continue
            cases.append(dict(name=f"audiomcq_{kind}",audio_path=row["audio_path"],question=row["question"],
                              choices=row["choices"],answer_index=row["answer_index"],expected_answer=answer,
                              source=row["source_dataset"],sample_id=row["sample_id"]))
            selected.add(kind)
            if len(selected)==3:
                break
    print("Loading original frozen Samsone-356M for native diagnostic",flush=True)
    encoder=FrozenSamsone(device=args.device)
    outputs=[]
    state_for_adapter=None
    action_choices=None
    for case in cases:
        captured={}

        def capture(module,inputs,kwargs,output):
            if "hidden" not in captured:
                captured["hidden"]=output.last_hidden_state.detach().cpu().clone()
                for key in ("inputs_embeds","attention_mask","position_ids"):
                    value=kwargs.get(key)
                    if value is not None:
                        captured[key]=value.detach().cpu().clone()

        hook=encoder.transformer.register_forward_hook(capture,with_kwargs=True)
        started=time.monotonic()
        try:
            response=encoder.helper(audio=case["audio_path"],prompt=case["question"],max_new_tokens=args.max_new_tokens)
        finally:
            hook.remove()
        mask=captured["attention_mask"]
        reference=encoder._last_valid(captured["hidden"],mask)
        with torch.no_grad():
            replay=encoder._forward(captured["inputs_embeds"].to(encoder.device),mask.to(encoder.device))
        state=encoder.state([case["audio_path"]],[case["question"]])
        silence=encoder.state([case["audio_path"]],[case["question"]],silence=True)
        projected=captured["inputs_embeds"]
        result={**case,"device":args.device,"native_response":response,"latency_s":time.monotonic()-started,
                "native_prefill_shape":list(captured["hidden"].shape),"valid_tokens":int(mask.sum()),
                "native_reference_replay_max_abs_error":(reference.float()-replay.float()).abs().max().item(),
                "native_vs_extraction_max_abs_error":(reference.float()-state.float()).abs().max().item(),
                "native_vs_extraction_relative_l2_error":((reference.float()-state.float()).norm()/reference.float().norm()).item(),
                "native_vs_extraction_cosine":F.cosine_similarity(reference.float(),state.float()).item(),
                "correct_vs_silence_hidden_l2":(state.float()-silence.float()).norm().item(),
                "correct_vs_silence_hidden_cosine":F.cosine_similarity(state.float(),silence.float()).item()}
        if case["name"]=="official_asset_water":
            waveform,rate=sf.read(case["audio_path"],dtype="float32")
            silence_path=audio_dir/"silence.wav"
            sf.write(silence_path,torch.zeros_like(torch.from_numpy(waveform)).numpy(),rate,subtype="FLOAT")
            result["native_response_on_silence_same_prompt"]=encoder.helper(audio=silence_path,prompt=case["question"],max_new_tokens=args.max_new_tokens)
        if "choices" in case and state_for_adapter is None:
            state_for_adapter=state
            action_choices=case["choices"]
        outputs.append(result)
        write_json(directory/"native_cases.json",dict(encoder_provenance=encoder.provenance,cases=outputs))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    checkpoint=ROOT/"runs/stage2/best.pt"
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model=AudioCLM(saved["spec"]["samsone_width"],bottleneck=saved["spec"]["bottleneck"])
    model.load_state_dict(saved["model_state"],strict=True)
    model.to(args.device).eval().requires_grad_(False)
    actions=encoder.action(action_choices)
    adapter_inputs=[]
    handle=model.adapter.register_forward_pre_hook(lambda module,inputs:adapter_inputs.append(inputs[0].detach().cpu().clone()))
    with torch.no_grad():
        zs=model.state(state_for_adapter.to(args.device))
        za=model.action(actions.to(args.device))
    handle.remove()
    path_proof=dict(state_adapter_receives_exact_extracted_hidden=torch.equal(adapter_inputs[0],state_for_adapter.float()),
                    action_adapter_receives_exact_extracted_hidden=torch.equal(adapter_inputs[1],actions.float()),
                    same_adapter_module_for_both_branches=True,adapter_forward_calls=len(adapter_inputs),
                    transformer_type=encoder.transformer.__class__.__name__,extraction="last_hidden_state at last valid token after final transformer norm",
                    state_shape=list(state_for_adapter.shape),action_shape=list(actions.shape),
                    adapter_input_width=model.spec["samsone_width"],adapter_output_width=model.spec["input_width"],
                    state_output_shape=list(zs.shape),action_output_shape=list(za.shape),
                    frozen_samsone=all(not p.requires_grad for p in encoder.model.parameters()))
    assert path_proof["state_adapter_receives_exact_extracted_hidden"] and path_proof["action_adapter_receives_exact_extracted_hidden"]
    write_json(directory/"adapter_path_proof.json",path_proof)
    print(json.dumps(path_proof),flush=True)


if __name__=="__main__":
    main()
