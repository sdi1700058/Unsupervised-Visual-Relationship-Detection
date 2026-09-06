# M4 — does latent distance track the number of actions between states?

**Hypothesis, in one sentence.** The number of observed transitions between
two latents grows one-for-one with the number of frames between them in a code
built from ground-truth positions, and grows more slowly, and then stops
growing at all, in a code a FOSAE learned.

Everything below the line marked *pre-registered* was written before the
metric was run on any export.

---

## Why this exists

`SPEC.md` V35 records that the H14 latent **compresses time**: frames 7 apart
in the video sat 4 to 6 transitions apart in the trained latent, against 7 for
the oracle. That was a **diagnosis of one clip of one run**, quoted as four
medians. It has never been a metric, so it cannot be compared across datasets
or across configurations, and its spread has never been reported at all.

Two of the methods indexed in `RELATED_WORK.md` name the same quantity and
optimise it directly. **P1 (Minimum Action Distance)** learns an embedding
where distance *is* the action count, from state trajectories alone. **P2
(temporal distances / contrastive successor features)** argues that a distance
without the triangle inequality *"translates to an inability to generalize and
find shortest paths"*. M4 measures on this project's own exports the quantity
those two papers optimise.

**What M4 is not.** It describes a code. It does **not** predict planner error,
and `SPEC.md` V36 records what happened the last time a temporal measure was
shipped as a predictor: across 8 clips of one model it correlated **+0.238**
with `mse_ratio`, the wrong sign. Nothing here may be used to rank two models
by expected planning quality.

---

## What is measured

For every ordered pair of frames `(i, j)` with `i < j` **inside one clip**:

| | |
|---|---|
| **frame gap** | `frame(j) - frame(i)`, from the export's own `frame_ids` where it carries them, and from the array index where it does not |
| **latent distance** | the fewest **observed transitions** that lead from latent `i` to latent `j` through the clip's own transition graph — the minimum action distance of `RELATED_WORK.md` P1 |

Pairs never cross a clip boundary. A multi-clip export concatenates clips, so a
pair spanning the join would compare two frames that are not separated by any
amount of time at all.

The graph is directed and is built per clip, because a transition observed in
another clip is not a transition available in this one.

Reported per group of clips:

- **`spearman`** and **`pearson`** between frame gap and latent distance.
- **compression ratio** `distance / gap`, per pair: the **median**, the
  **interquartile range**, the mean, the standard deviation, and the fraction
  of pairs preserved exactly. The spread is reported always. V35 quoted four
  medians and no spread, and a median alone cannot tell a code that compresses
  everywhere from a code that is exact on half its pairs and dead on the rest.
- **saturation.** The median distance curve `m(k)` is measured for every gap
  `k` up to `--max-gap`. Its local slope over a window of `w` gaps is
  `(m(k + w) - m(k)) / w`. The curve is monotonic while that slope stays at or
  above `--slope-floor` (default **0.1** transitions per frame — still gaining
  one transition per ten frames). `saturation_gap` is the first gap where the
  slope falls below the floor, and `None` when it never does.
- **`hamming_spearman`**, the same correlation against plain Hamming distance
  rather than transition count. It needs no graph, so it still reports when the
  transition graph collapses.

Saturation is the part that matters for planning. Once latent distance stops
growing with the frame gap, a planner cannot tell a long horizon from a short
one, and it will report that it reached the goal before the window is filled.

---

## Pre-registered reading

Checked **in this order**. The first row is a trap-catcher and comes first.

| condition | conclusion |
|---|---|
| fewer than 2 distinct latents, or `spearman < 0.30` | **SILENT.** Latent distance carries no temporal information on this data, so the compression ratio below is not a measurement of compression — it is a measurement of noise, and must not be quoted as a ratio. |
| `ratio_iqr >= 0.40` | **The median must not be quoted alone.** The ratio is not a property of the code on this dataset; it varies too much from pair to pair. Report the quartiles. |
| `ratio_median >= 0.95` | **Time preserved.** One latent transition per frame of real time. |
| `0.50 <= ratio_median < 0.95` | **Time compressed.** |
| `ratio_median < 0.50` | **Time severely compressed.** More than half of every horizon is lost. |

and independently, on the same run:

| condition | conclusion |
|---|---|
| `saturation_gap is None` | The relationship stays monotonic across the whole tested range. A planner can separate a long horizon from a short one anywhere inside it. |
| `saturation_gap` present | Beyond that gap the latent cannot express a longer horizon. This is the failure V35 points at, stated as a number. |

### What each outcome would mean for invariant V35, decided now

- **SUPPORTS V35** if, on VidVRD, the trained exports sit clearly below the
  oracle exports on `ratio_median`, and the oracle sits near 1.00.
- **CONTRADICTS V35** if the trained exports reach `ratio_median >= 0.95` on
  the same dataset V35 was measured on, or if the oracle compresses about as
  much as the trained code does. Either would make the ratio a property of the
  dataset rather than of the code, and V35 attributes it to the code.
- **NARROWS V35** if the VidOR oracle compresses while the VidVRD oracle does
  not. Compression would then be partly a property of how fast the filmed
  objects move against the bin grid, and V35's wording, which is already scoped
  to VidVRD, could not be widened beyond it.
- **WIDENS V35** if both oracles preserve time and both are separated from the
  trained code by the same margin. V35 currently rests on one dataset, which
  `DESIGN_WORKPLAN.md` section 4.4 caps at evidence strength 0.5.

A result that does not fall in any of those is reported as not deciding
between them.

---

## The command

Runs locally. numpy and the standard library, no model, no GPU, no video
frames. Memory-capped inside the script: an uncapped run crashed this
workstation on 2026-08-28.

```bash
bash experiments/M4_temporal_distance/run_local.sh
```

It builds the two oracle datasets if they are absent (about 4 minutes the first
time, nothing on later runs), then scores them beside H14's trained arms.

The metric alone, on exports that already exist:

```bash
python3 tools/planner/m4_temporal.py \
    vidvrd-oracle=eval/probe/batch \
    vidor-oracle=eval/probe/vidor \
    eval/exports/H14-P10-150010.npz \
    --max-gap 20 --out-dir eval/M4
```

A positional argument may be a single export, a directory of exports pooled
into one group, or `label=path` to name the group in the table and the figure.

**Expected runtime.** Under two minutes for the scoring, on top of the one-off
dataset build.

**Outputs**, all under `eval/`, which is not in git:

    eval/M4/m4_temporal.json
    eval/M4/m4_temporal.svg

---

## Results

Nothing above this line was changed after the run. **Measured** 2026-09-05,
`--max-gap 20`, on this workstation. Scoring took **9.7 seconds** with both
oracle datasets already built, against the two minutes expected.

Figure: `eval/M4/m4_temporal.svg`. Numbers: `eval/M4/m4_temporal.json`.

| group | clips | pairs | spearman | pearson | steps per frame | IQR | exact | saturates |
|---|---|---|---|---|---|---|---|---|
| VidVRD oracle | 20 | 25500 | +0.874 | +0.878 | **1.000** | 0.85 – 1.00 | 63.2% | no |
| VidOR oracle | 25 | 63002 | +0.787 | +0.795 | **0.889** | 0.62 – 1.00 | 40.0% | no |
| V35 clip, oracle | 1 | 1890 | +0.979 | +0.977 | **1.000** | 0.86 – 1.00 | 56.4% | no |
| V35 clip, trained P10 | 1 | 1890 | +0.955 | +0.948 | **0.900** | 0.73 – 1.00 | 30.7% | no |
| trained `U40 A2 P10`, 88 clips | 88 | 124194 | +0.492 | +0.497 | **0.300** | 0.09 – 0.58 | 8.5% | no |
| trained `U40 A2 P5`, 88 clips | 88 | 124194 | +0.446 | +0.448 | **0.167** | 0.00 – 0.38 | 3.3% | **gap 3** |
| trained `U20 A2 P10`, 88 clips | 88 | — | — | — | — | — | — | — |
| trained `U40 A2 P20`, 88 clips | 88 | — | — | — | — | — | — | — |

### Reading it by the rules above, in their order

- **`U20 A2 P10` and `U40 A2 P20` are SILENT**, and for the hardest reason
  available: each exports **one distinct latent across all 8610 frames**, every
  bit zero. Measured directly from the export. There is no distance to
  measure, so no ratio is reported for either.
- **`U40 A2 P10` trips the spread rule.** Its interquartile range is 0.49,
  above the 0.40 bar, so its median of 0.30 must not be quoted alone. On this
  dataset the ratio is not one property of the code: a quarter of pairs sit at
  0.09 or below and a quarter at 0.58 or above.
- **Everything else reads as pre-registered**: the two oracles and the V35 clip
  preserve or nearly preserve time, and both trained arms scored over all 88
  clips compress it heavily.

### V35's own table, reproduced exactly

Median latent transitions between frames *k* apart, on clip
`ILSVRC2015_train_00150010`, the clip V35 was measured on:

| k | 2 | 4 | 7 | 10 |
|---|---|---|---|---|
| oracle, V35 | 2 | 4 | 7 | 10 |
| **oracle, M4** | **2** | **4** | **7** | **10** |
| trained P10, V35 | 2 | 4 | 6 | 9 |
| **trained P10, M4** | **2** | **4** | **6** | **9** |

M4 shares no code with the measurement V35 came from beyond the export reader
and `observed_graph`, so this is a reproduction rather than a restatement.

### What this does to invariant V35

**It supports V35 and it narrows it, both by the rules written above.**

- **Supports.** On VidVRD the oracle sits at 1.000 and every trained arm that
  is alive at all sits below it, by a factor of 3.3 for `P10` and 6.0 for
  `P5`. V35's direction holds, and its own clip reproduces exactly.
- **Narrows, as the third bullet of the pre-registration allows for.** The
  **VidOR oracle compresses**: 0.889 pooled, and per clip a median of 0.889
  with a range of 0.333 to 1.000. Preserving time is therefore **not a
  property of a ground-truth positional code as such**. It is a property of
  how far the filmed objects move against the bin grid, and VidOR's move less.
  V35's wording is already scoped to VidVRD and this measurement says it may
  not be widened.
- **It adds the number V35 could not give.** V35 measured the P10 arm on one
  clip and got 0.86 at *k* = 7. Over the model's own 88 training clips the
  same arm gives **0.300**. The headline clip is about three times more
  faithful than the dataset it was drawn from, which is what `SPEC.md` V36
  already suspected from a different direction.
- **It adds saturation, which V35 had no way to state.** `P5` stops gaining
  distance at a frame gap of **3**. Its median distance reaches 3 transitions
  at gap 20 and stays there. A planner given that code cannot tell a 4-frame
  horizon from a 20-frame one.

Nothing here is evidence about planner error. `SPEC.md` V36 forbids that
reading and this experiment does not test it.

### A side observation, not part of the pre-registration

`hamming_spearman` — the same correlation against plain Hamming distance
instead of transition count — is **+0.337 for the VidVRD oracle**, against
+0.874 for the transition count, over the same pairs. Hamming distance is a
weak proxy for elapsed time even on a code that tracks time perfectly. That
is relevant to `Export.boxes_for`, which decodes an unobserved planner state
by falling back to its Hamming-nearest observed latent. Treat it as a
**pointer, not a result**: it is measured over gaps of 20 frames or less,
where `latent_geometry`'s `spearman` is measured over all pairs, so the two
numbers are not comparable and this one has not been checked against anything.

### Honest limits

- **Two datasets, one of them with no trained model.** VidOR is scored at the
  oracle only. Training a model on VidOR needs the cluster.
- **The frame gap is the array index** for the oracle datasets, which carry no
  `frame_ids`. Both datasets were built with `--no-fill`, which drops
  unannotated frames. Measured on the clips that compress most, annotation
  covers **100%, 99.5% and 100%** of frames, so index and frame number agree to
  within half a percent and the substitution changes nothing here. On a sparse
  dataset, such as Action Genome, it would not be safe.
- **Sources are sub-sampled**, at most about 120 to 240 per clip, so a long
  clip does not dominate the pool by length alone. The cap is approximate
  because the stride is an integer.
- **20 VidVRD clips and 25 VidOR clips.** Small, and the same screened lists
  the other experiments use, so they inherit whatever those screens select for.
