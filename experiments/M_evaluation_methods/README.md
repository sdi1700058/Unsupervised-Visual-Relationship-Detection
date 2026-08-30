# M1, M2, M3 — three evaluation methods that are not frame interpolation

The project had one evaluation method and needed more. These three ask
questions the interpolation metric cannot, and each one is scored against a
control that can make it report **silence** rather than a result.

Run them:

```bash
bash experiments/M_evaluation_methods/build_oracle_corpus.sh   # once
bash experiments/M_evaluation_methods/run_local.sh             # the ceiling
bash experiments/M_evaluation_methods/run_local.sh eval/exports/*H14*.npz
```

Local, numpy and the standard library, about a minute, memory-capped.

---

## Why three, and why these three

| | asks | needs ground truth? |
|---|---|---|
| interpolation (`mse_ratio`) | can a planner rebuild the frames between two endpoints? | **dense boxes** |
| **M1** predicate probing | does the latent encode the relations a person would name? | relation labels |
| **M2** compositional generalisation | is a predicate a rule, or tied to its objects? | relation labels + categories |
| **M3** plan validity | could this sequence have happened at all? | **none** |

M3 is the one that changes what data is usable. Every other method needs
densely annotated in-between frames, and that single requirement is what
restricts the thesis to 88 of 800 VidVRD clips and rules out VideoNet
entirely.

---

## M1 — does the latent encode the relations?

Fits two probes from the binary latent to VidVRD's own `relation_instances`
labels. **Ridge** is linear: it asks whether a relation is *expressed* in a
form a simple downstream reader can use. **kNN** asks only whether the
information is *present*.

Whole clips are held out. Within one clip most predicates never change, so a
per-clip split makes the probe and its control both score near 1.000.

### Pre-registered reading

`lift = ridge_mAP − max(prior_mAP, shuffled_mAP)`

| lift | conclusion |
|---|---|
| **≤ 0.02** | The latent carries no relational information a linear reader can use. If `knn_mAP` clears the bar while ridge does not, the relations are **present but not expressed** — a weaker and different claim, and it must be reported as that rather than as a pass. |
| **0.02 – 0.10** | Weak but real. |
| **≥ 0.10** | The latent carries relations a linear reader can recover. |

The bar is the **prior**, not the shuffle. Measured: scrambled latents scored
0.076 against a prior of 0.119, because scrambled latents are active noise and
therefore worse than no latent. Beating them proves nothing.

### Result on the ceiling — oracle latents, 20 clips, 6 held out

| | mAP |
|---|---|
| linear probe | 0.116 |
| kNN probe | 0.118 |
| shuffled control | 0.076 |
| **label prior** | **0.119** |
| control task (Hewitt & Liang) | 0.039 |
| **lift** | **−0.003** |
| **selectivity** | **+0.078** |

The **control task** answers the objection the other controls cannot: *was the
probe simply too weak?* Random labels fixed per latent type can only be fitted
by the probe's own memorisation, and the ridge probe scores 0.039 on them. It
is a **selective** probe that still cannot read the relations, so the negative
is about the representation rather than the instrument.

**A code built from ground-truth boxes carries no readable information about
VidVRD's relation labels.** This is a statement about the dataset, not about
FOSAE: `play`, `chase` and `touch` are not determined by where boxes sit in a
single frame.

It is a **ceiling**, so every purely positional model sits under it. FOSAE
sees appearance as well as position, so **a trained model scoring above 0.119
here would be reading something the boxes alone do not contain** — the
strongest positive result available from this experiment.

---

## M2 — a rule, or memorised objects?

FOSAE is a *First-Order* State AutoEncoder, and first order means a predicate
abstracts over its arguments. M2 tests that directly: hold out whole object
**categories**, keep the predicates, and compare against a size-matched
**random** split.

A predicate only counts if it is seen in training on some category *and* in
test on a held-out one. A predicate occurring only with the held-out category
tests novel labels, not composition, and is excluded.

### Pre-registered reading

Checked **in this order**, because the last row is a trap:

| condition | conclusion |
|---|---|
| `max(comp, random) − prior ≤ 0.02` | **NO SIGNAL.** Neither split beat the base rate, so the probe learned nothing on either and this says nothing about composition. A naive difference test would report "it generalises" here, which is the worst available reading. |
| `random − comp ≤ 0.02` | The predicate **transfers** to unseen object categories. First-order in practice, not only in name. |
| `0.02 < drop < 0.10` | Partial transfer. |
| `drop ≥ 0.10` | **Memorisation.** The predicates are tied to the object types they were trained on. |

### Result on the ceiling — holding out `bird`

| | mAP |
|---|---|
| compositional split | 0.168 |
| random split | 0.178 |
| label prior | 0.241 |

**NO SIGNAL**, as the first row predicts and as M1 implies: the probe never
worked on oracle latents, so the two splits agree for the wrong reason. M2
becomes informative only once a model clears M1's bar.

---

## M3 — could this have happened at all?

Four ways a decoded plan betrays itself, none needing a reference: an object
moving further in one step than anything seen in training; an object appearing
or vanishing mid-plan; a box with `x2 < x1`; a box off the canvas.

Thresholds are read off the export's own real transitions at a high
percentile — **measured, never chosen**. A constant would encode a guess about
how fast dogs run.

### Pre-registered reading

`separation = validity(real) − validity(scrambled)`, where scrambled is a
permutation of the same frames.

| condition | conclusion |
|---|---|
| `separation < 0.05` | **SILENT.** The measure cannot tell a real trajectory from a scrambled one on this data, so its validity score is not evidence. Checked first. |
| `validity ≥ 0.95` | Admissible. |
| `0.7 – 0.95` | Partly admissible; the rest break physics the training frames never broke. |
| `< 0.7` | Not admissible. This sequence could not have happened. |

### Result — six exports

| export | validity | separation | reading |
|---|---|---|---|
| `oracle-150010-fixed` (screened) | 0.989 | **+0.167** | real resolving power |
| `oracle-real-00005005` (56% invented) | 1.000 | **+0.000** | **SILENT** |
| trained P10, same bad clip | 0.967 | +0.008 | near-silent |
| trained P20, same bad clip | 0.933 | +0.025 | near-silent |
| `00040001` | 1.000 | +0.039 | weak |
| `00058003` | 1.000 | +0.128 | good |

**The clip that fails the control is exactly the one whose annotation
`--fill-annotations` fabricated.** Carrying the last box forward makes every
ordering equally plausible, so scrambling changes nothing. M3 found that from
a direction it was never pointed in, having no knowledge of the fill problem.

A metric that detects a known defect it was not designed for is a metric worth
trusting elsewhere.

---

## What these do not do

- **M1 and M2 measure the representation, not the planner.** A model can score
  well here and still plan badly.
- **M3 measures the decoded plan**, so a low score can come from the planner,
  the latent, or the decoder. Its value is being available where nothing else
  is, and that a plan failing it is definitely wrong whatever the real frames
  held.
- None of the three replaces the interpolation metric. They surround it.

## Honest limits of the current numbers

- 20 clips, 6 held out. Small.
- Every number above is on **oracle** latents. No trained model has been
  scored on any of the three, because H14 is still running.
- M2's held-out category is `bird`, chosen because it is the largest category
  in the batch that leaves a usable training set. A different holdout may give
  a different answer, and the experiment should be repeated across several.

---

## Results on a TRAINED model — added 2026-08-30

All three were originally run on the oracle only. They are now paired against
H14's trained FOSAE (`U40 A2 P10`, 400 bits) on ten random clips from its own
training set, identical frames.

| | oracle | trained |
|---|---|---|
| **M1** linear probe mAP | 0.116 | 0.098 |
| **M1** label prior (the bar) | 0.119 | 0.119 |
| **M1** selectivity | **+0.078** | **−0.003** |
| **M1** `attribute` tier lift | **+0.263** (2 of 2) | +0.060 (1 of 2) |
| **M2** verdict | NO SIGNAL | NO SIGNAL |
| **M3** decoded step / frame | **2.50 px** | **9.23 px** |
| **M3** clips it can judge | 10 of 10 | 1 of 10 |

Figure: `eval/summary/four_methods.svg`.

### How to read each of these, using the pre-registered rules above

- **M1.** Neither clears the label prior, so neither encodes VidVRD's relations
  readably. **Check selectivity before the tier table**: at −0.003 the probe
  fits random labels on the trained latent as well as real ones, so its
  per-predicate numbers are not interpretable. A single coupled predicate
  clearing 0.10 there is what a non-selective probe produces, not a finding.
- **M2.** NO SIGNAL on both, as the first row of its table predicts. It cannot
  speak until a model clears M1's bar. That is a limitation of M2 on this data,
  not a result about composition.
- **M3.** Reports SILENT on 9 of 10 trained clips. **The silence is not the
  evidence** — the cause is: the trained model's decoded boxes move 3.7x more
  per frame than the real ones, so the learned bound is 7.5x wider and a
  scrambled trajectory does not exceed it.

### What all three agree on, and what none of them show

Three different questions, no shared machinery beyond the export, same
conclusion: the trained representation is worse than a positional code at
everything measured, and the positional code is itself only adequate.

None of them is held out — every clip is inside H14's training set, so the
model fails on data it was fitted to. A held-out number needs a fresh export
from the cluster.
