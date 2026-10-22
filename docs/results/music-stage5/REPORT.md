# Musical Stage5

Frozen official Samsone356M → shared linear adapter960→4096 → original CLM state/action heads → normalized512vectors. Both heads and adapter train; encoder never runs inside training. No Qwen model.

100424train QA: NSynth38112, Jamendo50689, AudioMCQ-music replay11623. Instrument/artist/PCM groups do not cross splits. NSynth official validation/test overlap by instrument, so those held-out instruments were repartitioned; official train is preserved. Jamendo3-annotator unanimous full-track labels projected onto30s excerpts are weak labels, recorded as such.

Overfit512 first high constant LR reached89.45%/CE0.263 and failed the gate. The retained retry uses OneCycleLR, adapter3e-4/heads1e-4,2048updates; train100%/CE0.00195. Full Stage5 starts from raw common, rather than memorized diagnostic weights.

Full training: batch256, AdamW, OneCycleLR, max32epochs/patience5; adapter max LR1e-4, both heads5e-5, scale1e-3; candidate CE + verified pair CE×0.5 + swap×1, cosine margin0.05.128pairs per update, with diverse different-source partners. Stopped25epochs/9825updates/492.6seconds; selected20epochs/7860updates by class-balanced task accuracy on fixed validation. Both original heads changed, encoder_calls0. Calibration fits one positive scale on validation only,84.43→59.76, preserving rankings.

| Fixed test2044QA | Top1 | CE | ECE |
|---|---:|---:|---:|
| Correct audio |76.52%|0.6655|0.0364|
| Random different source audio |39.19%|2.2563|0.3614|
| Silence through the encoder |41.10%|1.9408|0.3082|

Audio gaps+37.33/+35.42pp. Equal-task balanced music accuracy82.11%; balanced audio gaps+37.38/+37.26pp. Macro selection uses tasks with≥50validation examples; all tasks, including sparse genre taxonomies, remain in the detailed reports.

Instrument family57.58% on561unseen-instrument QA; instrument source55.73% on192QA. Several mood/vocals cohorts exceed90%, but genre cohorts have only22–49QA. Do not describe rare genres, all musical tasks or real-world10s windows as uniformly solved. Training/evaluation use complete supported excerpts; web applies the specified padded10s window.

Live Python Ranker checked15test files across tasks, maximum probability difference0.000174 versus cached evaluation. Artifacts, frozen features and both diagnostic attempts returned locally and SHA-verified before Stage6. Immutable common is preserved. Raw best SHA `21006be9f0de3dee0fe783797945bb7dd6cc1d005b6c49b57c6bb33f2043510b`.

Stage5 curves (historical image omitted from this text-only export)
