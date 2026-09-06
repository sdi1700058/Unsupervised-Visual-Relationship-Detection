# G6 — why every run lands on the same number

**Ready to paste. Step 0 needs no GPU. The training half needs Sherlock.**

## The hypothesis, in one sentence

Every video run so far has been sitting on the analytic floor of its own loss
function, and three changes this fork made against upstream are what put it
there.

## Why this experiment exists

The overnight sweep varied `ZEROSUPPRESS` (off and 0.2), `MAX_TEMPERATURE`
(0.5 and 5.0), `U/A/P` (10/2/8 and 20/2/16), the batch size, the transition
mode, four clips and two whole categories. Sixteen runs. **Every result landed
between 0.488 and 0.597.** MNIST, on the same code, reaches `1.038e-07`.

When no knob moves the number, the varied knobs are not the cause.

They were not. **`val_loss = 0.5245` is `H(0.2182)`** — the binary entropy, in
nats, of the mean of the data. That is exactly what a decoder pays when it
ignores its input and emits one constant. The loss is BCE averaged over the
whole flattened feature vector (`latplan/util/distances.py`), so its floor is
the density of the data and carries **no dependence on `U`, `A` or `P`**.
Different architectures converging on the same number is then not a
coincidence and not a plateau. It is the closed-form answer for a model that
has stopped using its latent.

Measured on CPU over 12 VidVRD videos at 30fps:

| max objects | patch | pad share | data mean | floor `H(p)` |
|---|---|---|---|---|
| 3 | 32 | 26.5% | 0.3337 | 0.6368 |
| 4 | 32 | 41.2% | 0.2679 | 0.5812 |
| **5** | **32** | **52.4%** | **0.2173** | **0.5235** |
| 3 | 8 | 26.5% | 0.1881 | 0.4835 |
| 5 | 8 | 52.4% | 0.1256 | 0.3780 |

The row in bold is the sweep's configuration, and 0.5235 against an observed
0.5245 is the whole mystery.

Two independent checks agree. `tools/planner/liveness.py` reports **4 of 21
exports** holding one distinct code for every frame. And the observation is
consistent with the one video configuration that does learn:
`PREENC_LAYERS=2 PREENC_DIM=1000` reaches `val 0.12`, far below any floor in
that table, because the pre-encoder takes 3272 dimensions down to 1000 and
breaks the raw pixels' monopoly on the gradient.

## The three suspects

All three are this fork's changes against upstream, all three are annotated
`was X` in `strips.py`, and all three push toward the same degenerate optimum.

| # | knob | upstream | here | why it points this way |
|---|---|---|---|---|
| A | `zerosuppress` | **0.0 (off)** | 0.05 | `model.py:780` adds `alpha * K.mean(encoder.output)` to the loss. A **linear** penalty has a constant gradient, so it never stops pushing the latent toward zero. The comment says "try user's idea" — it is a guess with no measurement beside it |
| B | `max_temperature` | **5.0** | 1.0 | annealing runs 1.0 → 0.7, which is nearly flat. The Gumbel is near-discrete from the first epoch, so gradients are tiny before anything is learned |
| C | `zerosuppress_delay` | **0.1** | 0.05 | A applies from the 5th percent of training, when the encoder has learned nothing yet |

B is worth stating carefully: the code's own comment claimed 1.0 "keeps the
Gumbel latent near-discrete from epoch 1" as though that were the fix, twenty
lines from `min_temperature: 0.7` annotated "never gets near-discrete". Both
cannot be true. The hypothesis recorded in the comments — that a high early
temperature was the cause — is the opposite of what the arithmetic says.

## Step 0 — the diagnostic, before any GPU time

`tools/planner/collapse_floor.py` computes the floor from a baked dataset with
numpy alone. Run it on every dataset and run you already have. It costs
seconds and it tells you how many past runs were dead before you spend a night
finding out.

```bash
# On Sherlock, where the baked npz files are.
cd "$THESIS"

# The floor of each baked dataset.
python3 tools/planner/collapse_floor.py data/npz/video/vidvrd/overfit/*.npz

# A specific run against the floor of the data it trained on.
python3 tools/planner/collapse_floor.py \
    data/npz/video/vidvrd/overfit/<the-npz>.npz \
    --run out/video/vidvrd/<the-run-dir>
```

It prints the data mean, the floor, and one of three readings: **at the
floor** (the latent is carrying nothing), **below the floor** (the model is
using it), or **above the floor** (worse than predicting the mean, which is
its own kind of broken).

**Read step 0 before submitting anything below.** If the existing runs are not
at the floor, this hypothesis is wrong and the arms should change.

## The arms

One dataset, one latent shape, one seed. Only the named knob moves. The
baseline repeats the current configuration so the comparison is against a
number produced by the same code on the same day, not against a remembered
one.

| arm | change | prediction if the hypothesis holds |
|---|---|---|
| **base** | nothing | `val_BCE` within 0.01 of the floor |
| **A** | `ZEROSUPPRESS=0.0` | below the floor |
| **B** | `MAX_TEMPERATURE=5.0` | below the floor, or unchanged |
| **C** | `ZEROSUPPRESS_DELAY=0.1` | between base and A |
| **D** | `PREENC_LAYERS=2 PREENC_DIM=1000` | well below the floor, near 0.12 |

**D is a positive control, not a candidate.** It is the configuration already
known to learn. If D does not come out below the floor, the measurement is
broken rather than the hypothesis, and nothing else on this page should be
believed.

## The readings, pre-registered

Written before the runs, so the result cannot be reinterpreted afterwards.

| what happens | what it means |
|---|---|
| base at the floor, A below it | `zerosuppress` is the cause. Revert it to upstream's 0.0 and re-run the sweeps that were flat |
| base at the floor, B below it, A not | the temperature schedule is the cause, and the comment in `strips.py` has the sign backwards |
| A and B both below, C between | the two compound; the delay decides how much |
| every arm at the floor, D below | none of the three is the cause. The pre-encoder is doing something the three knobs cannot, and the next question is the 93.9 percent of the gradient the raw patch block takes |
| D at the floor too | the measurement is wrong. Stop and fix step 0 |
| base **below** the floor | the premise is wrong. The sweep was not collapsed and this whole page is void |

## What this does not test

**The patch size.** It is held fixed on purpose. The patch and the box block
are concatenated with no weighting and the loss averages over the result, so
the patch size sets the share of the gradient that position gets: 51.0 percent
at patch 8, 20.7 at 16, 6.1 at 32, 2.8 at 48
(`tools/planner/collapse_floor.py::box_share`). A patch-size sweep is therefore
also a loss-reweighting sweep, and its arms cannot be compared to each other as
though only the input had changed. That is a separate experiment and G5's
results need re-reading in that light.

**Whether a model that clears the floor can plan.** Clearing the floor means
the latent carries something. Whether what it carries is *plannable* is what
the evaluation grid measures, and it comes after this.

## Cost

Five arms, one clip, same epoch budget as H14. Nothing here is a sweep.
