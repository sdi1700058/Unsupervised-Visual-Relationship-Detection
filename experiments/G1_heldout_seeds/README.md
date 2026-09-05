# G1 — held-out clips, and three seeds per arm

**Status: ready to run. The training half needs Sherlock.**

## The hypothesis, in one sentence

A FOSAE model trained on 70 VidVRD clips plans on 18 clips it never saw as
well as it plans on 18 clips it did see, and the difference between the two is
larger than the difference between three training runs of the same
configuration.

## The two gaps this closes

**Gap 1 — nothing in this project is scored out of sample.** Every FOSAE model
so far trained on the 88 clips in `eval/vidvrd_winnable_clips.txt` and was then
scored on those same 88 clips. A planner that reaches a goal in a clip the
model memorised tells us nothing about the representation.

**Gap 2 — every headline rests on one training run.** `strips.py` carries no
seed knob, so no margin in any document has ever been compared against the
spread of repeated runs. A 2x margin between two arms means nothing until the
spread between three runs of one arm is known.

## The design

### The split, and how to re-derive it

88 screened clips, split by clip id into 70 train and 18 held out. The split is
a hash of the clip id under a fixed seed string, so it depends on no random
number generator and no Python version:

```
seed string  G1-heldout-2026-09-05
key(clip)    sha1("G1-heldout-2026-09-05:" + clip_id)
order        clips sorted by key, ascending
test18       order[0:18]
train70      order[18:88]
seen18       order[18:36]      a fixed subset of the training clips
```

Re-derive both lists with:

```bash
python3 - <<'PY'
import hashlib
ids = [l.strip() for l in open('eval/vidvrd_winnable_clips.txt') if l.strip()]
key = lambda c: hashlib.sha1(("G1-heldout-2026-09-05:" + c).encode("utf-8")).hexdigest()
order = sorted(ids, key=key)
print("test18 ", ",".join(order[:18]))
print("train70", ",".join(order[18:]))
print("seen18 ", ",".join(order[18:36]))
PY
```

Both lists are **inlined** in `run_sherlock.sh`. `eval/` is excluded from
version control, so `eval/vidvrd_winnable_clips.txt` never reaches the cluster,
and a script that reads it there aborts. That defect bit `sh/h14.sh` once and
the fix is written into its comments.

The script also checks the two lists do not intersect before it bakes anything,
and that every `seen18` clip is one of the 70. An overlap of one clip would
destroy the experiment in silence.

`ILSVRC2015_train_00150010`, the clip the oracle reached 14 of 14 windows on
and the clip most earlier headlines rest on, falls in the **training** split.
So this experiment does not re-score it out of sample, and a reader comparing
G1 against those headlines is comparing different clips.

### The files

| file | what it does | where it runs |
|---|---|---|
| `run_sherlock.sh` | checks the split, bakes three npz files, submits three seeds, chains the export | Sherlock, one `sbatch` |
| `export_sherlock.sh` | encodes every seed twice, on `seen18` and on `test18` | Sherlock, chained by `--dependency=afterany` |
| `score_local.sh` | checks replicate independence, screens the geometry, slices, plans, then summarises | the workstation |
| `g1_summary.py` | the reading, the table and the figure | called by `score_local.sh` |

`export_sherlock.sh` exists because `sh/export_model.sh` always encodes the npz
a model trained on, and it names its output after the model directory. G1 needs
the same model encoded against a **different** bake, twice, under two names.
`export_latents.py --npz-path` has supported that all along and nothing in the
project has used it.

### Why there is a third bake, `seen18`

The obvious comparison is `train70` against `test18`. It is the wrong one,
because the preprocessing is **data dependent**. `latplan/puzzles/util.py`
runs `equalize_hist` over the whole array of patches and then rescales by the
global minimum and maximum. A bake of 70 clips and a bake of 18 clips therefore
map the same pixel to different values, so a `train70`-against-`test18`
comparison mixes generalisation with a preprocessing shift.

`seen18` removes that. It holds 18 clips the model **did** train on, baked on
its own so its preprocessing sample size matches `test18`. The comparison the
experiment reports is `seen18` against `test18`:

| | `seen18` | `test18` |
|---|---|---|
| clips | 18 | 18 |
| the model trained on them | yes | **no** |
| bake size, so preprocessing statistics | 18 clips | 18 clips |
| planner operators, from `actions.csv` | the 70 training clips | the 70 training clips |
| frames on disk at 30 fps | 2,156 | 1,901 |
| windows at K=16, stride 15, capped at 3 per clip | 54 | 53 |

The frame and window counts are **derived** from the frame directories on the
workstation. The bake keeps annotated frames only, so the counts after baking
will be a little lower.

The planner operator set comes from `actions.csv` in the model directory, which
holds the training transitions. It is identical in both arms. So a held-out
window asks the right question: can the operators learned on 70 clips reach a
goal state taken from a clip the model never saw?

### Three seeds

Three training jobs, one configuration, three output trees:

```
out/G1/seed1/video/vidvrd/...
out/G1/seed2/...
out/G1/seed3/...
```

`OUT_DIR` is the supported knob for this, and `latplan/util/paths.py` names the
reason: each tree gets its own `grid_search.log`, and a shared log with
`LIMIT=1` makes the second and third run short-circuit instead of training.
Three jobs at one `OUT_DIR` would also fight over one `net0.h5`.

**What varies between the three runs, stated honestly.** `strips.py` accepts no
seed, so the three runs are **unseeded replicates**: TensorFlow picks its own
initialiser seed when no graph seed is set, so the three runs start from
different weights. That is *inferred* from the TensorFlow 1 seeding rule, not
measured. `score_local.sh` therefore hashes the latents of the three exports
and prints a loud warning if any two match bit for bit, because identical
latents would prove the replicates are not independent and would void the
seed half of this experiment.

A single replicate is therefore not reproducible bit for bit. Making it
reproducible needs a seed knob inside `strips.py`, which this experiment does
not touch.

### The window

Scored at **window 16, not 8** (`SPEC.md` V37). The crossover criterion that
selected these 88 clips is a steep function of window size, and at window 8 the
oracle itself loses to a straight line on 4 of 4 random screened clips, while
at window 16 it wins on 6 of 10. Scoring at 8 measures something the selection
never promised. Override with `WINDOW=` only with a reason.

Every export is cut into per-clip exports with `tools/planner/slice_export.py`
before any planning. A window that straddles a clip boundary pairs the last
frame of one video with the first frame of another, and every windowing tool in
this project slides blindly along the concatenated frame axis.

## What each outcome means — decided before the run

Two numbers decide the reading, and both come out of `score_local.sh`:

- **drop** = median across seeds of the `seen18` solve rate, minus the median
  across seeds of the `test18` solve rate.
- **noise** = the larger of the two seed spreads, where an arm's spread is its
  highest solve rate across the three seeds minus its lowest.

The rule, fixed in advance: **a drop counts only when it is larger than the
noise.**

| result | reading | what follows |
|---|---|---|
| `drop` is inside `noise`, and the held-out `mse_ratio` sits inside the seed spread of the seen arm | FOSAE generalises to unseen VidVRD clips at this scale | Every earlier headline can be restated as a held-out number. Gap 1 closes and the 88-clip results stand as they are written. |
| `drop` is larger than `noise`, and the held-out arm still solves some windows | The representation carries part of the way, and part of the earlier score was memorisation | Relabel every earlier headline as **in-sample**, and report the held-out number beside it. The margin against the oracle shrinks by the measured drop. |
| the held-out arm solves near zero windows while the seen arm solves many | The learned operator set does not reach states from clips the model never saw. The code is clip specific | A measured negative with a named cause, and a strong result to report. The next experiment asks whether more clips or fewer bits fixes it, and neither is assumed. |
| the held-out arm scores **better** than the seen arm | Something other than generalisation drives the numbers | Do not celebrate it. The 18 held-out clips are easier by chance, so re-run the split under a second seed string before writing anything down. |
| `noise` is as large as any difference between the arms | One run per arm was never enough to support a margin | Every single-seed headline in the project needs three seeds before it is reported again. This is a result about the method of measurement, and it changes what may be written. |
| both arms solve zero windows, or `val_loss` stays above about 0.4 | The model did not train at 70 clips | Nothing here measures held-out anything. Record the cost in the unit's `attempts` list and give the next attempt more epochs and more memory. One failure is a cost, not a verdict. |

The last two rows are the ones to expect. The best result in this project so far
came from a full-batch overfit on one clip.

## How the summary reports the numbers

`score_local.sh` follows the house rules that `tools/planner/e1_summary.py`
records, and it adds one:

1. **Solve rate leads.** Reaching a goal does not depend on how far the boxes
   move, so it is the axis least confounded by clip difficulty.
2. **Error is compared as `mse_ratio`, never as raw `bbox_mse` across arms.**
   The two arms run on different clips, and `SPEC.md` V38 measured that raw
   planner error is dominated by how non-linear a clip happens to be. Raw error
   appears in the table, marked as not comparable across arms.
3. **An arm that reaches under half its windows has its errors labelled
   optimistically biased.** The windows it reached are its easiest.
4. **Distinct windows are counted, never CSV rows.** A row is one window scored
   by one method, and two methods run here. Row counting doubled every E1 figure
   once already. The new part: a window key is `(clip, init, goal)`, not
   `(init, goal)`, because each arm holds 18 per-clip exports and frame index 0
   exists in every one of them.

Where both methods scored one window, the arm is credited with the better of
the two rather than with both.

## Running it

**On Sherlock.** One command, then walk away. The export chains itself on with
`--dependency=afterany`, so a dead arm does not withhold the arms that lived:

```bash
cd $SCRATCH/panos/sgg-thesis && git pull && mkdir -p logs && sbatch experiments/G1_heldout_seeds/run_sherlock.sh
```

Watch it with `squeue -u $USER`. When the queue empties, push the six exports:

```bash
git add -f eval/exports/G1-seed*.npz && git commit -m "G1 exports" && git push
```

**On the workstation**, after pulling those exports:

```bash
git pull && bash experiments/G1_heldout_seeds/score_local.sh
```

That writes `eval/planner/G1_summary.md` and `eval/planner/G1_summary.svg`.

## Expected runtime

| stage | budget in the script | expectation |
|---|---|---|
| bake, 70 + 18 + 18 clips | 12 h | E1 baked 22 clips inside a 4 h ceiling, so 106 clip-bakes need room |
| train, 3 seeds in parallel | 10 h each | H14 gave 6 h to 88 clips at 3,000 epochs; this has 70 |
| export, 6 exports | 2 h | minutes of work; the budget covers the queue |
| score, on the workstation | no limit | up to about 3.5 h in the worst case, and usually under 2 |

The worst case assumes every one of the 642 planner calls uses its whole 20
second budget. Lower it with `BUDGET=`, `MAX_WINDOWS=` or `METHODS=pddl`.

## Resource caps

`score_local.sh` caps every planner process at `ulimit -v 6000000`, which is 6
GB. An unbounded search over a wide latent exhausted memory and took this
workstation down on 2026-08-28.

## What was verified, and what was not

| check | result |
|---|---|
| `bash -n` on all three shell scripts | pass |
| `g1_summary.py` parses under the 3.6 rules, and holds no walrus, no f-string `=` and no future annotations | pass |
| `g1_summary.py` produces the right reading on synthetic rows, both directions | pass |
| all 88 clip ids have annotations under `data/video/vidvrd/annotations/train` | pass, 0 missing |
| all 88 clip ids have frames under `data/video/vidvrd/frames_30fps/train` | pass, 0 missing |
| the union of the inlined train and test lists is exactly the 88 ids in `eval/vidvrd_winnable_clips.txt` | pass |
| the three inlined lists reproduce the documented hash split element by element | pass |
| train and test do not intersect, and `seen18` is a subset of train | pass, and the script re-checks all three on the cluster. Each guard was made to fire by hand |
| the split guards abort with status 2 and bake nothing when a clip is in both lists | pass |
| `slice_export.py` cuts a synthetic multi-clip export into one file per clip, and no window at K=16 crosses a boundary | pass |
| `latent_geometry.py` accepts six export paths and `--csv` | pass |
| the replicate-independence check reports both a clash and a clean set | pass, both branches were made to fire |
| `export_latents.py` accepts `--npz-path`, which is what makes held-out export possible | pass, read from the parser |
| the feature dimension does not depend on the clips in a bake | pass, read from `common/encode.py`: the patch size and the box grid are both fixed |
| `score_local.sh` behaves when the exports are absent | pass, it prints what to run |

**Not verified: the bake, the training and the export.** `setup-dataset.py`
imports `config.py`, which imports TensorFlow, and TensorFlow is absent from the
sandbox. So the argument names come from reading the parser and the commands
have never run here. A failed bake aborts before any training job is submitted,
so the failure mode is a wasted queue slot rather than a wrong number.

**Not verified: that the three replicates differ.** The reasoning is in the
"Three seeds" section above and the check runs inside `score_local.sh`.
