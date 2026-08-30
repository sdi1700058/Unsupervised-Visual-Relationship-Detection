# E1 — Does innate structure predict plannability?

**Status: ready to run. Needs Sherlock for the training half.**

## The hypothesis, in one sentence

A FOSAE model trained on clips whose subject is **rule-governed** produces
latents a planner can use, and one trained on clips of equal size, motion and
difficulty but **without rules** does not.

This is `DATASETS.md` Criterion 0 made falsifiable. Until now it has been an
argument, and a persuasive one, but nothing has tested it.

## Why it matters

Criterion 0 currently outranks annotation density in the dataset strategy on
the strength of reasoning alone. If it is real, the dataset search should be
reorganised around it and the robotics corpora move to the front. If it is not,
that reorganisation would be a mistake and the effort belongs elsewhere.

## Design — a paired, matched comparison

Eleven clips per arm, each unstructured clip matched to a structured one by
motion. Every pair agrees on motion to within 0.5 px per step.

| | structured (A) | unstructured (B) |
|---|---|---|
| clips | 11 | 11 |
| transitions | 1,204 | 874 |
| median motion | **10.7 px/step** | **10.6 px/step** |
| median frames | 75 | 60 |
| median objects | 2 | 2 |
| median crossover | 0.411 | 0.337 |
| **median coupled coverage** | **1.00** | **0.00** |

All clips are fully annotated, so `--fill-annotations` is **off** and no
transition is fabricated. All are arithmetically winnable, so `mse_ratio < 1`
is reachable in both arms.

### The known confound, stated up front

**Arm A has 38% more transitions** (1,204 against 874), because some
motion-matched clips are longer. That difference favours A, which is the
direction of the hypothesis. So:

> **A win for A smaller than roughly 38% is inconclusive**, not a positive
> result. Say so if it happens.

Everything else that could plausibly matter is matched. The remaining
difference between the arms is structure.

## What each outcome means — decided before running

| result | reading | what follows |
|---|---|---|
| **A's planner error materially below B's** (well past the 38% margin) | Criterion 0 is real and operative at this scale | Reorganise the dataset search around structure. Robotics corpora move to the front. |
| **A ≈ B** | Structure does not predict plannability at this data volume | Criterion 0 is not disproven, but it stops being the organising principle. Look at volume and model capacity instead. |
| **B better than A** | The criterion is backwards | Take it seriously. Re-examine the predicate classification in `screen_vidvrd.PREDICATE_TIERS` first, since it is a judgement. |
| **Both fail to train** (`val_loss` above ~0.4) | ~1,000 transitions is too few, whatever the structure | Neither arm says anything about Criterion 0. Rerun at the full 74-clip / 6,391-transition scale. |

That last row is the likeliest single outcome and is worth expecting: the best
result so far came from a full-batch overfit on one clip.

## What has been verified, and what has not

Checked before handing this over, because a silent failure costs a whole
Sherlock cycle:

| check | result |
|---|---|
| both scripts parse (`bash -n`) | pass |
| `sweep_lib.sh` sources and defines `bake`, `submit`, `section`, `sweep_totals` | pass |
| every flag used exists in `setup-dataset.py`'s parser | pass — read from the argparse definitions |
| all 22 video ids exist in `annotations/train` | pass |
| all 22 have frames in `frames_30fps/train` | pass |
| `score_local.sh` behaves when the exports are absent | pass — prints what to run |
| `latent_geometry.py` accepts two explicit files | pass |

One thing looked wrong and was not: six ids carry an `ILSVRC2015_val_` prefix,
which reads like a different split. It is ILSVRC's original naming, and those
83 files live inside VidVRD's **train** split. The loader takes
`split="train"` and finds them.

**Not verified: the bake itself.** `setup-dataset.py` imports `config.py`,
which imports **TensorFlow** at line 11 — not matplotlib, which is present.
TensorFlow is a large install and PyPI is unreachable from the sandbox, so the
bake cannot run here at all. So the argument names are confirmed by reading the parser
but the command has never run. That is the residual risk, and it shows up
immediately — `bake` prints `BAKE FAILED <name>` and the arm is skipped rather
than producing a wrong result.

## Running it

**On Sherlock** — about 40 minutes wall for both arms:

```bash
cd $SCRATCH/panos/sgg-thesis && git pull
mkdir -p logs && sbatch experiments/E1_structure/run_sherlock.sh
```

Then, once the jobs finish:

```bash
git add -f eval/exports/*E1-*.npz && git commit -m "E1 exports" && git push
```

**Locally**, after pulling those exports:

```bash
bash experiments/E1_structure/score_local.sh
```

That prints the comparison and writes `eval/planner/E1_summary.md`.
