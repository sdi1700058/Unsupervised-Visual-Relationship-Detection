# G3_collapse_rerun — is the collapse a property of the configuration, or of the draw?

## Hypothesis

**The seven dead exports come from run-to-run instability rather than from the
latent shapes that produced them, so a re-run of the same five configurations
collapses far fewer than five times.**

## Why the question is open

`tools/planner/liveness.py` reported **7 of 31 exports dead** on 2026-09-18,
each with `distinct=1`, which means the encoder emitted one latent for every
frame. Sixteen exports came from one bake, one set of defaults and one sweep.
Seven collapsed and nine did not, and no axis separates them:

| | P5 | P10 | P20 |
|---|---|---|---|
| U5 | **dead** | live, 427 | live, 487 |
| U10 | — | **dead** | live, 3341 |
| U20 | — | **dead** | **dead** |
| U40 | live, 1856 | live, 3023 | **dead** |
| U80 | live, 1401 | live, 163 | — |

A property of the latent shape would draw a line somewhere on that table. This
one does not: `U5 P5` dies while `U5 P10` lives, `U10 P10` dies while `U10 P20`
lives, and `U40 P20` dies while `U80 P10` lives on a third of its distinct
count.

Seven exports come from **five** trained models. `U20_A2_P10` and `U40_A2_P20`
each produced two exports, one on the 88-clip set and one on clip
`ILSVRC2015_train_00150010`.

This is gap **G3** in `notes/NOW.md`, stated generally: one run per
configuration, no seeds, no error bars.

## The command

Preflight first. It writes nothing, submits nothing, and refuses for a named
reason:

```bash
cd $SCRATCH/panos/sgg-thesis
bash experiments/G3_collapse_rerun/preflight.sh
```

Then one command, and walk away:

```bash
cd $SCRATCH/panos/sgg-thesis && mkdir -p logs
sbatch experiments/G3_collapse_rerun/rerun_collapsed.sh
```

The export job is chained with `--dependency`, so there is no second visit.
When `squeue --me` is empty, the number this turns on:

```bash
.venv-local/bin/python tools/planner/liveness.py --exports eval/exports --strict
```

## What each outcome means, decided in advance

Let **k** be how many of the five re-trained configurations collapse again.

| k | reading | what follows |
|---|---|---|
| **0 or 1** | The collapse is the draw, not the configuration | The seven reds carry no information about latent shape. Every single-run export in this project is one sample from a distribution nobody measured, which is exactly what G3 says. The next step is seeds, not a new architecture |
| **2 to 3** | Both matter, and neither is separable at n=1 | Do not name a shape unusable. The honest next step is three seeds for each of the five, which is fifteen jobs and still cheap on a queue |
| **4 or 5** | The collapse is the configuration | Those latent shapes do not train under these defaults. Record which, with the count, and stop exporting them. A `SPEC.md` invariant becomes worth writing |

**The gate reading is a separate number and it does not decide this.** Dead
exports fall from 7 by however many re-runs live, and `export liveness` goes
green only at k = 0. A green gate is not the finding, and a still-red gate at
k = 1 is not a failure.

## Expected runtime

| | |
|---|---|
| jobs | 5 training, plus 1 chained export |
| walltime requested | 6 h for the three arms at 200 bits and below, 10 h for the two at 400 bits and above |
| export | 1 h, chained `afterany` so a dead arm does not withhold the others |
| wall clock, if the queue is free | about 10 to 11 h |
| bake | none. The dataset exists and the script refuses rather than baking |

## The limits, written before the run

**No seed is pinned anywhere in this project.** `grep -rn "SEED\|seed"` over
`workbench/sh/sweep_lib.sh`, `workbench/sh/submit.sh` and `strips.py` returns
nothing on 2026-09-18. That is what makes a re-run a genuinely fresh draw, and
it is also why **this experiment is not reproducible**: a third run would be a
third draw. Reading k as a rate needs a seed knob and three runs for each arm,
and that is the work this experiment argues for rather than the work it does.

**n is five, and one run each.** k is a count, not a rate. The table above is
written so that no row claims more than a count can carry.

**The re-run overwrites in place.** A run directory is keyed by a hash of its
parameters, so an identical configuration writes to the identical path. The
script moves the five collapsed directories to
`out/video/vidvrd/collapsed-<date>/` first, so the before half survives.
