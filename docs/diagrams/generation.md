# Audio System-1 diagrams

Generated with the built-in imagegen tool on 2026-10-03. These images describe the corrected v2 architecture and pipeline; they do not imply completed stages or a passed audio-use gate.

The following specifications describe the historical corrected-v2 diagrams.

## Inference prompt

Create a crisp, readable technical infographic titled “Audio System-1 — inference architecture”. Show two separate state/action paths: audio + question and choice text pass through one frozen Samsone-356M with shared weights. Each hidden vector has width 960. Both paths use the same Linear adapter, with shared weights θ, mapping 960 to 4096, followed by L2 normalization. State and action then pass through their distinct original CLM heads, whose weights are initialized from the official checkpoint and subsequently retrained. Normalize each 512-dimensional output. Compare z_state and z_action by cosine, multiply scores by checkpoint scale 100, softmax across choices, and return a ranking with probabilities. No generation, no Qwen, no alternative model. Keep parallel representations separate; never merge hidden vectors. Use a light neutral background, precise thin arrows, clear text, muted frozen blocks, purple adapter blocks, blue heads, and teal output.

## Training prompt

Create a matching readable technical infographic titled “Audio System-1 — training pipeline”. Show preparation: AudioMCQ strong records, canonical dataset, immutable audio-identity splits and nested train subsets, frozen Samsone-356M state/action feature extraction, FP32 computation and BF16 sharded feature storage. Training must read stored features without running Samsone again. Show overfit on 1000 examples with adapter trainable and both CLM heads frozen, a decreasing-loss gate, Stage 1 on 10K with the same freeze policy, and a bounded Stage 2 pilot on 10K with adapter and both heads trainable. Both pilot and the main Stage 2 start from best Stage 1. The audio-use gate requires correct-audio accuracy to exceed random-audio and silence accuracy by at least five percentage points. A failed gate leads to diagnostics and a pilot using verified swap-loss pairs, followed by another gate check; it cannot bypass the gate. A passed gate allows Stage 2 on 50K, Stage 3 on 150K, and Stage 4 on all 181592 usable training samples. These three stages train adapter and both heads, initialize from the previous main stage's best checkpoint, and require an audio gate before expansion. Show best.pt, last.pt, metrics.json and validation metrics Top-1, Top-2, CE, mean correct probability, ECE for music, sound, speech, temporal and overall. Samsone remains frozen throughout. Do not imply a gate has already passed.

## Training image correction

Preserve the existing style, labels and layout. Redirect the precompute connector to the beginning of the training sequence at overfit. Remove any diagnostic-to-Stage-3 shortcut. Route diagnostics back to the audio-gate check. Add “Start: best Stage 1” to both the bounded Stage 2 pilot and main Stage 2.
