# Training and evaluating Audio System 1

Historical pipeline report; use the current README and v3 recipe for the final
configuration. Original CLM heads are pretrained initialization. Stage 1 freezes
both heads while aligning the adapter. From Stage 2, adapter, state_head, and
action_head train together. Frozen Samsone is excluded from the optimizer and
is not even instantiated by the training process. Both branches share one
adapter object.

The mechanical gate passed on 1000 examples: CE 1.385 → 0.381, train Top-1 99.2%.
This permitted further training. Equal correct/random/silence accuracy on the
early adapter-only checkpoint failed to establish audio dependence; it did not
justify keeping heads frozen in subsequent stages.

Main Stage 1 used 10K train and 1K each validation/test. Validation Top-1 grew
from 21.7% to 62.8%. A separate diagnostic Stage 2 on the same 10K confirmed
nonzero gradients and changes to both pretrained heads. It is not the main
50K Stage 2; that run initializes from the best main Stage 1 checkpoint.

Supervisor `scripts/run_pipeline.py`:

1. Uses the immutable 10K snapshot, creates/resumes frozen features and Stage 1.
2. Fixes an immutable 50K snapshot, computes features, and trains Stage 2 from
   the best Stage 1.
3. Waits for available source preparation, fixes the full canonical corpus,
   and verifies at least 150K training examples.
4. Builds the full feature cache by reusing existing representations, then
   trains Stage 3 on 150K and Stage 4 on the complete available training split.
5. Evaluates best Stage 4 on untouched test and compares real-WAV live ranking
   with cached inference.

Phase and heartbeat are written to `runs/pipeline_status.json`. Checkpoints
store optimizer/RNG, epoch order/offset, exact feature-manifest hash, encoder
provenance, and training audio identities. Resume requires the same stage and
feature corpus. A subsequent stage can use a larger corpus but checks that old
training audio does not overlap new held-out audio.

Historical objective: cross entropy over literal cosine scores. AdamW updates
the adapter and both heads in Stages 2–4 with separate learning rates.
Validation selects best; test is used for final evaluation. Training subsets
are selected by deterministic hash order.

Audio ablation replaces the recording with another from the same split while
preserving question/candidates. Silence traverses the same encoder. Correct
references use exactly the same evaluation cohort. Missing precomputed
ablations are explicitly marked unavailable.

The operational audio-dependence threshold in this report requires Top-1 to
fall by at least 5 percentage points for both random audio and silence. Also
inspect category counts and predictions. If final accuracy barely changes
when audio is replaced, the Audio System 1 hypothesis is not supported even
when training loss falls. Record that as an unsuccessful experiment while
preserving the architecture.

A working API, decreasing training loss, and high training accuracy do not by
themselves establish a successful model.

Live encoding uses a different padded batch size from precompute; BF16
representations can differ numerically. On a tested SpeechCraft WAV,
cached/live Top-1 matched and maximum probability difference was 0.00756.
Inference verification records the actual error, Top-1 agreement, and absolute
tolerance 0.01. That numerical tolerance is not the audio-dependence criterion.

The same choice texts encoded in batches of 4/128 had relative L2 difference
0.0453 in BF16 versus 0.00000845 in FP32, using the same right padding and
original frozen transformer. Large-run features stay BF16 in memory; only
the selected minibatch is cast to FP32 at the adapter input, preserving the
computation while reducing memory use. Later v3 extraction uses FP32 compute.
