# G4 — the grid over U and P

**Ready to paste. The training half needs Sherlock.**

## The hypothesis, in one sentence

The latent shape `(U, P)` changes how well a FOSAE latent plans, and `U = 40`
was never chosen on evidence, so a narrower latent may plan better while it
reconstructs worse.

## Why this experiment exists

`U` is the number of predicate units and `P` is the number of predicates. The
latent holds `U * P` bits. **`U` has never been moved.** Every trained-model
number in this project comes from `U = 40` or `U = 20`. `P` has moved a little:
H14 ran P5, P10 and P20, always at `U = 40` or `U = 20`.

The value 40 also looks too large on the model's own terms. `latplan/model.py`
line 1442 builds one Gumbel-softmax over the object axis for each of the
`U * A` attention slots. The bake uses `--max-objects 3` and every arm uses
`A = 2`, so a unit can select 3 by 3, which is **9 distinct argument
bindings**. At `U = 40` the model holds 40 units over 9 possible bindings. The
ladder below puts `U = 5` just under that number and `U = 10` just over it.

Latent size is also a planning decision. Fast Downward searches `2^(U*P)`
states, and H14 measured 17 of 19 windows solved at 400 bits against 3 of 19 at
800 bits on one clip. Nothing in the reconstruction loss shows that.

## The grid

One dataset, one configuration, one window. Only the latent shape moves.

| | P = 5 | P = 10 | P = 20 |
|---|---|---|---|
| **U = 5** | 25 bits | 50 bits | 100 bits |
| **U = 10** | 50 bits | 100 bits | 200 bits |
| **U = 20** | 100 bits | *H14* | 400 bits |
| **U = 40** | *H14* | *H14* | *H14* |
| **U = 80** | 400 bits | 800 bits | *left out* |

*H14* marks a cell H14 already trained on this exact `npz`. G4 skips it and
does not repeat it. `U80 P20` is left out on purpose: 800 bits already collapses
planning, and 1600 bits doubles the exponent of the search space for the cell
that costs the most and tells the least.

**10 new cells. 14 cells in the figure**, because the four H14 cells join it at
no extra cost. G4 reuses H14's bake and H14's stem, so every export carries the
name `..._catH14-winnable88-30fps-mo3-nofill-p8_...` and one glob collects all
14. If H14's run directories are gone from the cluster, the figure draws those
four cells as `none` and says so.

### What is held fixed

The data is H14's: the 88 screened clips, fully annotated, 8,522 real
transitions, `--fill-annotations` **off**. The clip list is inlined in
`run_sherlock.sh`, because `eval/` is gitignored and a read from
`eval/vidvrd_winnable_clips.txt` would abort the job on Sherlock.

The configuration is H14's: `EPOCH=3000 LR=0.001 BATCH=1000 PREENC_LAYERS=2
PREENC_DIM=1000 MAX_TEMPERATURE=1.0 TRANSITION_MODE=sequential A=2`. That is
the one that learned, at `val 0.1216`.

**Every cell asks for identical resources: 48 GB, 10 hours, 1 GPU.** A cell
with a smaller budget could stop early and train for fewer epochs. The grid
would then measure the budget and not the latent shape. `G4_train.csv` records
the epoch count per cell, so a short cell shows up in the table with a warning
instead of passing as a result.

## Run it

**On Sherlock. One command, then leave.**

```bash
cd $SCRATCH/panos/sgg-thesis && git pull
mkdir -p logs && sbatch experiments/G4_up_sweep/run_sherlock.sh
```

That submits **12 SLURM jobs in total**: 1 driver, 10 training jobs, and 1
export job. The export is chained with `--dependency=afterany`, so it runs
itself when the last training job ends. `afterany` and not `afterok`, because a
cell that dies must not withhold the cells that lived.

When `squeue -u $USER` is empty, push what landed:

```bash
git add -f eval/exports/*catH14-winnable*.npz eval/exports/G4_train.csv
git commit -m "G4 exports" && git push
```

**Locally, after a pull:**

```bash
bash experiments/G4_up_sweep/score_local.sh
```

That writes `eval/planner/G4_summary.md` and `eval/planner/G4_summary.svg`.

### Expected runtime

| stage | expectation | worst case |
|---|---|---|
| bake | skipped, because H14 built the `npz` | 4 to 6 h if the `npz` is gone |
| training, per cell | under 6 h *(inferred: H14 budgeted 6 h for the same data at 400 bits)* | the 10 h wall |
| GPU hours | about 40 to 60 used | 100 reserved |
| export | minutes per cell | the 1.5 h wall |
| local scoring | minutes *(the measured median solve time on this task is 0.1 s)* | 4.7 h, if all 14 cells time out on all 20 windows |

Nothing in the local half needs a GPU. Every local process runs under
`ulimit -v 6000000`, because an unbounded planner run crashed this workstation
on 2026-08-28.

## Score at window 16, not 8

`score_local.sh` uses window 16. This is not a free parameter. The crossover
criterion the 88 clips were selected by is a steep function of window size:
one clip reads `floor/baseline` 2.50 at w=8, 1.06 at w=12, 0.48 at w=16 and
0.07 at w=32 (`SPEC.md` V37). Re-running the screen at window 8 keeps only 39
of the 88 clips. Scoring at 8 would measure something the selection never
promised. Override `WINDOW=` only with a stated reason.

## What each outcome means — decided before the run

Both directions, and every threshold below is also a constant in
`g4_summary.py`. Changing one after reading the data turns the experiment into a
search for a story.

| constant | value | what it is |
|---|---|---|
| `PLANNABLE_VAL_LOSS` | 0.5 | below this a model is plannable (`SPEC.md` C17) |
| `MATERIAL` | 1.5 | a difference in median `mse_ratio` under this factor does not count |
| `RHO_AGREE` | 0.5 | at or above this the two axes agree |
| `RHO_DISAGREE` | 0.0 | at or below this the two axes disagree |

### Question 1 — does `U` change planning quality?

The comparison is the median `mse_ratio` at `U <= 10` against the median at
`U >= 40`.

| result | reading | what follows |
|---|---|---|
| **low U better by 1.5x or more** | A narrow latent plans better. `U = 40` was never tested and it handicapped every earlier planning number | Rerun the decisive arms at the best `U`. Re-open the H14 negative result |
| **high U better by 1.5x or more** | Capacity binds before search width does | Extend the ladder upward to `U = 160` instead of downward |
| **inside 1.5x** | `U` is not the limiting parameter at this data volume | Stop tuning the latent shape. Look at the data and the objective |

### Question 2 — do the two axes agree?

Spearman rank correlation between the reconstruction number and the median
`mse_ratio`, over the cells that carry both.

| result | reading | what follows |
|---|---|---|
| **rho >= 0.5** | Reconstruction loss ranks the cells the way the planner does | Report it as a property of this grid, not as a refutation of `EVAL.md` 5.7. Model selection could then skip the planner, which is a large saving |
| **rho <= 0.0** | The two axes disagree. A cell reconstructs well and plans badly | `EVAL.md` 5.7 and the A4 argument, confirmed inside this project's own data. Never select a latent shape by training loss |
| **0.0 < rho < 0.5** | Weakly related | Neither claim holds. Do not use training loss to choose a cell, but do not claim it misleads |

### Question 3 — the shape, or only the size?

The grid holds three cells at 100 bits and three at 400 bits, with different
`(U, P)`. Their spread answers whether `U * P` describes the latent on its own.

| result | reading | what follows |
|---|---|---|
| **equal-bit cells differ by 1.5x or more** | The shape matters. `U` and `P` are two parameters | Report `U` and `P` separately everywhere. `U * P` is not a sufficient description |
| **equal-bit cells agree inside 1.5x** | Only the bit count matters | Collapse `U` and `P` into one knob for planning, and say which grid supports that |

### The two outcomes that answer nothing

| result | reading | what follows |
|---|---|---|
| **every cell reports `val_BCE >= 0.5`** | No cell is plannable by `SPEC.md` C17 | The grid says nothing about `U` or `P`. The limit sits upstream of the latent shape |
| **no cell reaches a single window** | Worse than the row above | Read `G4_train.csv` first. A loss that never moved means the runs produced nothing |

`score_local.sh` prints whichever of these rows applies. It does not choose
between them after the fact.

## The figure — both axes, because one axis hides the result

`eval/planner/G4_summary.svg` holds three panels.

1. **Reconstruction** over the `(U, P)` grid. Training loss when the cluster
   recorded it, and round-trip box error otherwise.
2. **Planning** over the same grid: median `mse_ratio`, with the reached-window
   count under each cell. A green border marks a cell that beat the straight
   line.
3. **The two panels against each other.** One point per cell. A dashed line
   marks `mse_ratio = 1`. The corner that holds cells which reconstruct well
   and plan worse than a straight line is shaded and labelled.

Panel 3 exists because `notes/docs/EVAL.md` 5.7 records that reconstruction
loss does not select for plannability. If the claim holds on this grid, the
shaded corner fills up and the rank correlation goes negative. If it does not
hold, the two heatmaps look alike. Either way a reader sees it in one glance.

## Two known limitations, stated before the run

**The window slider does not know about clip boundaries.** The export
concatenates 88 clips, and `tools/planner/common/windows.py` walks a frame
index. A window can therefore start in one clip and end in the next, where it
scores a cut instead of motion. `g4_summary.py` reads the export's `frame_ids`,
drops every row whose two endpoints sit in different clips, and prints the
count. `sh/h14_score.sh` does not do this, so a `mse_ratio` from G4 is not
directly comparable with one from H14.

**One dataset.** Every number here comes from VidVRD. Any ordering of the cells
is a hypothesis until a second corpus reproduces it, and the figure says so in
its footer.

## What has been verified, and what has not

Checked before handing this over, because a silent failure costs a Sherlock
cycle.

| check | result |
|---|---|
| every script parses (`bash -n`) | pass |
| the embedded Python parses and needs nothing above 3.6 | pass |
| `g4_summary.py` passes `tools/check_py36.py` | pass |
| `sweep_lib.sh` `submit` argument order, `<tag> <stem> <mem> <time> VAR=val` | pass, read from the function |
| `eval_plannability.sh` accepts every flag `score_local.sh` passes | pass, run end to end on a real export with `--stride` |
| `latent_geometry.py` accepts many files and `--csv` | pass, same call as `sh/h14_score.sh` |
| the export name carries `U`, `A` and `P` | pass, measured on `eval/exports/U40_A2_P10_catE1-structured...` |
| `export_facts` reads a real export with `allow_pickle` off | pass, round-trip box MSE 1744.4 on the E1 export |
| the clip-boundary filter keeps a within-clip window | pass, 0 of 4 rows dropped on a within-clip run |
| `score_local.sh` behaves when the exports are absent | pass, prints what to run and exits 2 |
| `g4_summary.py` on a 14-cell synthetic grid | pass, table, reading and figure all render |
| every branch of the pre-registered reading | pass, each one exercised on synthetic cells |
| `G4_train.csv` absent | pass, the figure falls back to round-trip box error |

**Not verified: the bake, the training and the export.** `setup-dataset.py`
imports TensorFlow, which is absent from this sandbox, so no cluster-side
command has run. The bake is also the step least likely to run at all, because
H14 already built the `npz` and `run_sherlock.sh` skips a bake it finds on disk.

**Not verified: whether H14's run directories still sit under
`out/video/vidvrd/`.** The export step globs them, and the four H14 cells join
the figure only if they are there. Their absence costs four cells and nothing
else.

**Not verified: the local scoring stage on real G4 exports**, because none
exist yet. The planner call itself ran end to end on the E1 export at window
16 with `--stride 60`, which is the same call with a different file.

One artefact was left behind by that check:
`eval/planner/G4-smoke-flagcheck/`. It holds 4 rows from the E1 export and no
G4 data. It does not match the `G4-U*-P*` glob, so it cannot enter the figure.
Remove it whenever you choose.
