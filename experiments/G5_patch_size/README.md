# G5 — patch size, held against everything else

**Ready to paste. The training half needs Sherlock.**
**Incomplete: there is no scorer. See "What is missing" at the foot.**

## The hypothesis, in one sentence

The 8×8 patch starves the image half of the loss, so a larger patch
reconstructs the frame better — and the question is whether it also plans
better, or whether the two come apart.

## Why this experiment exists

Every bake in this project uses `--patch-size 8`. The feature vector is
`[patch_size² × 3 | x1,y1,x2,y2 one-hot]` and the box block is fixed at 200
dimensions, so the patch alone decides the balance:

| patch | patch dims | box share of the loss |
|---|---|---|
| 8 | 192 | 51% |
| 16 | 768 | 21% |
| 32 | 3072 | 6% |
| 48 | 6912 | 3% |

At patch 8 the model spends half its capacity on four numbers per object. The
reconstructions are reported as unrecognisable, which is consistent with that
split, and the overfit runs that reconstructed acceptably used patch 48 and 64.
Nothing has ever measured the two against each other on the same data.

## The one thing that must be read correctly

**`val_loss` here is four quantities wearing one name.** Because the patch
decides that balance, it also decides *what the loss is*. Every arm minimises a
figure over a differently sized and differently weighted feature vector.

A lower number at patch 32 than at patch 8 is **not** a better reconstruction.
It is a loss in which the box block — the part that is hard to fit — has been
reweighted from about half the total down to a sixteenth. Each arm also has its
**own constant-solution floor**, the entropy of its own mean feature, and a run
sitting on that floor has learned nothing whatever the number reads. Compute it
per arm with:

```bash
python3 tools/planner/collapse_floor.py data/npz/video/vidvrd/overfit/G5-*.npz
```

So: **read `val_loss` down a column, never across the arms.** "Reconstruction
improves" can only mean one of

- the same arm improving against its own earlier run, or
- the decoded boxes and the decoded patches scored **separately**, in their own
  units, which is the only comparison the four arms share.

The planning axis is unaffected: `mse_ratio` is measured in canvas pixels
against a straight line, and that scale is identical in all four arms. It is
therefore the only number here that may be compared across arms directly.

## The arms

`PATCH_LIST="8 16 32 48"`, one bake each, over the same 88 screened clips G4
and H14 used, at 30fps, `--max-objects 3`, no fill. Everything except the patch
is held fixed. Memory scales with the patch, so each arm requests its own.

## What each outcome means, decided before the run

| outcome | reading |
|---|---|
| reconstruction improves **and** planning improves | patch 8 was a mistake, and every earlier planner number was taken on a starved model. Rebake |
| reconstruction improves **and** planning does not | reconstruction loss does not select for plannability — `EVAL.md` §5.7 measured directly rather than argued. A publishable negative |
| neither improves | the patch is not the limit; the limit is upstream in the data. Look at the bake, not the model |
| a larger patch fails to train at all | record the memory it needed and raise the request. One failure is a cost, not a verdict |

## Running it

```bash
cd $SCRATCH/panos/sgg-thesis && git pull
mkdir -p logs && sbatch experiments/G5_patch_size/run_sherlock.sh
```

Then, once the training jobs finish:

```bash
bash experiments/G5_patch_size/export_sherlock.sh
```

Four bakes, then four training jobs of about 40 minutes each at 3000 epochs,
run in parallel. Under two hours wall clock.

## What is missing

**There is no `score_local.sh`.** G1, G4 and E1 each have one; G5 does not, and
until 2026-09-07 `export_sherlock.sh` told the reader to run a file that has
never existed. Score the arms one at a time instead, and compare only
`mse_ratio`:

```bash
bash tools/planner/eval_plannability.sh eval/exports/G5-p8.npz  --window 8
bash tools/planner/eval_plannability.sh eval/exports/G5-p16.npz --window 8
# ... and so on, then read mse_ratio across the four
```

**Nothing reads `G5_train.csv`.** `export_sherlock.sh` writes it and no tool
consumes it.

This page exists because the pre-registration above lived only in a script
header, where nobody browsing the experiments would find it. The experiment is
sound and it is not finished; both facts belong here rather than in a comment.
