# Next iteration: musical genre generalization

This historical plan follows completed Stages 5/6; it did not itself launch
training. The architecture stays frozen Samsone356M + SAME adapter + both
original pretrained CLM heads.

## Measured problem

On a user-uploaded MP3 window, 158.71–168.71s, question `what is music genre?`,
and choices `rock`, `hip hop`, `classic instrumental`, the old checkpoint
reproduces `classic instrumental` at 59.58%. Silence with the same choices
produces `hip hop` at 95.58%. Audio changes scores, but predictions are unreliable.

With nine trained genres and `Which broad musical genre best describes this excerpt?`,
the old model gives `pop` 55.64%; with the same choices and the user's question,
it gives `rap hiphop` 44.52%. The lower-LR model likewise changes from rap hiphop
60.58% to pop 53.43% when wording changes; the original three-choice set still
selects classic instrumental 57.75%. This is one diagnostic recording, not a
new benchmark or proof of its genre label.

Official native Samsone356M describes pop for `Describe the music heard.`;
a voice question distinguishes the file from silence. However, `Caption the audio`
returns the uninformative `the audio`, and silence descriptions invent musical
properties. Full generated answers are retained in the private diagnostic and
are **not** gold genre labels or a reliable automatic teacher.

Of 190,653 training QA, new sources supplied only 1,483 explicit genre questions
on 793 unique audio identities: genre 527, genre_gtzan 334, genre_rosamerica 479,
electronic 143. The first taxonomy has pop 26, funk/soul/R&B 11, jazz 13.
Another 1,274 AudioMCQ music replay questions mention genre/style, but keyword
counts do not form an audited shared genre taxonomy. Small genre validation
cohorts are excluded from the main macro selector at its support threshold of 50.

[Diagnostic](results/music-stage6-window/user_genre_diagnostic.json),
[distributions](results/music-stage6-window/train_task_counts.json).

## Concrete plan

1. Fix a genre validation benchmark with checked labels on the actual 10s
   windows: several hundred clips, balanced classes, semantic question/answer
   paraphrases, and source/artist/PCM-disjoint splits. Test missing-correct-choice
   cases explicitly: closed candidate ranking selects a supplied option and
   does not automatically report a genre absent from the candidate set.
2. Expand genre-specific data. Revisit downloaded Jamendo human consensus,
   separating verifiable 2/3 majority from 3/3 agreement without hiding noise.
   Then add [MTG-Jamendo genre](https://github.com/MTG/mtg-jamendo-dataset):
   the official subset has 55,215 tracks / 95 genre tags. Uploader tags need
   a coarse taxonomy and audits of multilabel/conflicting annotations. Missing
   tags do not establish negatives. Count usable examples after filtering,
   rather than declaring 55K gold examples. FMA can supply another source
   after checking taxonomy, splits, and duplicates.
3. Prepare genuine 10s training windows. Windows from one source inherit its
   split and do not become independent recordings. Distinguish clip labels
   from weak full-track labels. Preserve provenance and exclude held-out
   recordings from training.
4. Diversify questions and choice-text aliases while preserving meaning and
   correct answers. Paraphrases are views of one QA; frozen Samsone precomputes
   each variant separately. Test simple user questions and trained templates.
5. Run a separate genre continuation with balanced task/class/source sampling
   and instrument/vocal/structure replay. Select the balance with a validation
   pilot: genre is initially primary, replay retains earlier capabilities.
   Pass overfit 512 on the new manifest before scaling. Select checkpoints
   using genre macro and wording robustness while retaining audio gaps and
   replay quality, then evaluate fixed test. Preserve the architecture and
   keep Samsone out of the training loop. More epochs on the old 190K mixture
   cannot substitute for these data.
