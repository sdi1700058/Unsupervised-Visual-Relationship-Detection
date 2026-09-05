# M7 — mean average precision on relation triplets

**Status: run 2026-09-05. Results are at the end.**

Everything above the "Results" heading was written before any number existed.
It is left unedited on purpose, so the pre-registration can be checked against
what happened.

This file states the hypothesis, the exact command and the meaning of every
outcome *before* any number exists. Read the "What each outcome means" table
before you read the results.

## The hypothesis, in one sentence

Scored by the VidVRD protocol itself, with ground-truth tubelets supplied, a
relation predictor read out of this project's oracle latent scores above the
no-vision frequency floor on VidVRD and on VidOR.

## Why it matters

`RELATED_WORK.md` section B holds the published ImageNet-VidVRD ladder, from
mAP 8.58 in 2017 to 31.33 in 2024, with 43.15 under VrdONE's oracle-trajectory
condition. `EVAL.md` §5.3 records that this project has never computed that
quantity, and that the M1 probe mAP is a *different* measurement: M1 averages
average precision over predicates on per-frame rows, and the field averages
average precision over *videos* on localised triplet instances. Supervisor
rubric B1 asks for the field's number. Until it exists, nothing measured here
can be placed beside a published row.

## The protocol, and where it came from

`tools/planner/m7_map.py` is a port of the dataset author's own toolkit:

- <https://github.com/xdshang/VidVRD-helper/blob/master/evaluation/visual_relation_detection.py>
- <https://github.com/xdshang/VidVRD-helper/blob/master/evaluation/common.py>
- <https://github.com/xdshang/VidVRD-helper/blob/master/dataset/dataset.py>
  (`get_relation_insts`, which defines what a ground-truth instance is)

The variant is the one the VidVRD and VidOR challenges score: **relation
detection** by mean average precision, where the mean is over videos and the
average precision is the all-point VOC rule; a prediction matches a
ground-truth instance when the triplet strings are equal and
`min(subject vIoU, object vIoU) >= 0.5`; matching is greedy by score and each
ground-truth instance is consumed once. **Relation tagging** is
Precision@1/5/10 over distinct triplets with localisation ignored. Recall@50
and Recall@100 are pooled over the corpus, not averaged per video. Boxes use
the inclusive-pixel convention, so a width is `xmax - xmin + 1`.

Two details decide whether a number is comparable, and both are copied rather
than chosen: average precision is averaged **per video and then over videos**,
and a video with no ground-truth relation is dropped rather than scored zero.

## The conditions

Every condition supplies **ground-truth tubelets**, so all of them sit in
VrdONE's oracle-trajectory regime (`EVAL.md` §5.4). Compare against the 43.15
row, never against 31.33.

| tag | predictor | what it establishes |
|---|---|---|
| `*-perfect` | the ground truth returned as the prediction | the harness is correct. **mAP must be exactly 1.000.** |
| `*-frequency` | the most frequent triplets of the training split, on every ordered pair of ground-truth tubelets | the no-vision floor. A model that sees nothing still scores this. |
| `*-probe` | the M1 ridge probe read off the oracle latent, cut into 30-frame segments and associated | what this project can claim today |

`*-probe` uses the oracle latent exports already on disk
(`eval/probe/batch`, 20 VidVRD clips; `eval/probe/vidor`, 25 VidOR clips), all
written by `oracle-bins60x40`. **No export on disk carries model-predicted
relations for either corpus**, so `*-probe` measures the *positional ceiling*,
not a trained FOSAE. The trained-model row needs a cluster run and is not
available here.

## The exact command

One command, from the repository root, no edits:

```sh
sh experiments/M7_map_triplets/run.sh
```

It writes `eval/M7/<tag>/map.json` and `eval/M7/<tag>/map.svg` for each
condition, then `eval/M7/m7_map.svg` comparing them.

## What each outcome means — decided before running

| result | reading | what follows |
|---|---|---|
| **`*-perfect` is not 1.000** | the harness is wrong | Report nothing else. Every other row is void. |
| **probe mAP above frequency mAP on both corpora** | the oracle latent carries relation information the field's own metric can see | The number goes on the ladder as an oracle-condition row, and M1's negative lift is a property of the per-frame framing rather than of the representation. |
| **probe mAP at or below frequency mAP** | the latent adds nothing the metric can see | Report the floor and the probe together and say so. This agrees with M1's measured lift of -0.003, and it strengthens rather than weakens the thesis claim that a purely positional code cannot express predicates such as `chase`. |
| **probe mAP above frequency on one corpus only** | the effect depends on the corpus | Do not average the two. Name the corpus in every sentence that quotes the number. |
| **frequency mAP near or above the published 2017 row (8.58)** | the ground-truth tubelets are doing the work, not the relation predictor | The oracle condition is generous. Say so beside every number, and prefer the gap between conditions to any single value. |

A further check that costs nothing: the frequency floor must be **well below**
the perfect condition and **well above** zero. A floor of exactly zero means
the predictions never localise, which is a bug in the association step and not
a result.

## Expected runtime

Estimated before running, on the workstation, single core:

| step | estimate |
|---|---|
| VidVRD, 200 test videos, perfect and frequency | about 2 minutes |
| VidOR, 200 validation videos, perfect and frequency | about 6 minutes |
| the two probe conditions | about 2 minutes |
| **total** | **about 10 minutes** |

Run it under `ulimit -v 6000000`. `run.sh` sets that itself.

## Results

Run 2026-09-05. **Measured**, from `eval/M7/*/map.json`. Figure:
`eval/M7/m7_map.svg`. Measured runtime 2 minutes 49 seconds, against the 10
minutes estimated above.

All figures are on the 0 to 100 scale the published tables use.

| condition | videos | mAP | R@50 | R@100 | P@1 | P@5 | P@10 |
|---|---|---|---|---|---|---|---|
| vidvrd-perfect | 200 | **100.00** | 69.43 | 84.69 | 100.00 | 100.00 | 100.00 |
| vidvrd-frequency | 200 | **30.33** | 19.09 | 24.18 | 70.00 | 52.90 | 40.37 |
| vidvrd-probe | 6 | **1.81** | 3.82 | 3.82 | 0.00 | 5.56 | 11.27 |
| vidor-perfect | 200 | **100.00** | 74.01 | 92.23 | 100.00 | 100.00 | 100.00 |
| vidor-frequency | 200 | **16.94** | 17.84 | 23.96 | 66.00 | 52.20 | 42.56 |
| vidor-probe | 7 | **3.92** | 4.12 | 4.12 | 71.43 | 57.14 | 45.71 |

### The harness is correct

`*-perfect` gives exactly 100.00 on both corpora, so the pre-registered
"report nothing else" condition does not apply.

A second check was not planned and is worth recording. `*-perfect` does **not**
give recall 100, and it must not: Recall@K keeps only the top K predictions of
each video, so a video holding more than K ground-truth relations loses the
excess. The arithmetic ceiling is `sum(min(K, relations)) / relations`.
**Measured**: 69.43 and 84.69 on VidVRD, 74.01 and 92.23 on VidOR — equal to
the recorded recalls to the last digit. The protocol is implemented as
published, including the part that looks like a shortfall.

### The floor is high, and that is the headline

The pre-registered row "frequency mAP near or above the published 2017 row"
fired, and hard. A predictor that **sees nothing** — the training split's most
frequent triplets, put on every ordered pair of ground-truth tubelets — scores
30.33 on VidVRD and 16.94 on VidOR. The published detection ladder runs from
8.58 to 31.33 on VidVRD, and the best VidOR row in `EVAL.md` §5.3 is 6.85.

So the pre-registered response applies without amendment:

> The oracle condition is generous. Say so beside every number, and prefer the
> gap between conditions to any single value.

The reason is measurable rather than mysterious. **Measured** on the first 5
VidVRD test clips: every clip has 2 to 4 tracks that persist for the whole
clip, and most ground-truth relations span the whole clip too, so a
whole-tubelet prediction reaches volumetric IoU 1.0 by construction. Handing a
system ground-truth tubelets on this corpus removes most of the localisation
problem, which is the same effect VrdONE measures from the other side
(`RELATED_WORK.md` B4, +11.82 on VidVRD and +29.89 on VidOR).

**No number in this table may be placed on the published ladder.** The ladder's
rows detect their own tubelets. These do not.

### The probe scores below the floor, on both corpora

`vidvrd-probe` 1.81 against a floor of 30.33, and `vidor-probe` 3.92 against
16.94. The pre-registered reading:

> the latent adds nothing the metric can see. Report the floor and the probe
> together and say so. This agrees with M1's measured lift of -0.003, and it
> strengthens rather than weakens the thesis claim that a purely positional
> code cannot express predicates such as `chase`.

Two facts stop this being read as a broken run.

- **The probe is M1's probe, on M1's split.** `vidvrd-probe` reports 20 clips,
  6 held out, 3,990 training rows and 58 predicates. `eval/probe/M1-oracle-corpus/probe.json`
  reports 20 clips, 6 held out, 3,990 training rows and 58 predicates. The two
  metrics describe one model, so M7 explains M1 rather than competing with it.
- **No export scored here is dead.** `n_dead_exports` is 0 for both, and the
  smallest export holds 35 distinct latent codes on VidVRD and 43 on VidOR.
  Four exports under `eval/exports/` hold one all-zero code each; `m7_map.py`
  drops such an export and records the count, so a number can never be quoted
  without it.

`reachable_fraction` is 0.885 on VidVRD and 0.763 on VidOR: the probe only ever
sees the three largest tracks, so relations on the rest cannot be found at all.
That caps the probe rows and it is reported rather than corrected.

### What is real and what is not

**Real, measured, needs no cluster:** the metric, the harness check, and both
floors. Those are properties of the two corpora and of the protocol.

**Real but narrow:** the probe rows. They are computed from oracle latents,
which encode ground-truth boxes, so they measure the **positional ceiling** —
what any purely positional code could reach — and not a trained FOSAE.

**Not available here:** a trained-model row. No export on disk carries
model-predicted relations for either corpus, and producing one needs a cluster
run. When it exists, `run.sh` scores it by adding one more `--predictor probe`
line with the new exports; nothing in `m7_map.py` has to change.
