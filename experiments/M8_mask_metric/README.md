# M8 — score a plan against masks, not only against boxes

**Hypothesis, in one sentence:** a mask is a usable object boundary for this
evaluation, because a mask metric with its own round-trip floor produces a
number comparable with the box results, and a tight box read off a mask makes
a mask dataset comparable with VidVRD, VidOR and Action Genome under every
metric already written.

**The general rule this serves.** How a dataset draws an object boundary is a
property of the **dataset**, and the evaluation adapts to it. The reverse rule,
where a dataset has to fit whatever evaluation happens to exist, is why usable
candidates kept being set aside. PVSG is the case in point: 150,000 labelled
frames over 400 videos with temporal scene graphs, passed over only because
its objects are masks.

---

## Read this first: what is NOT measured here

**PVSG is not downloaded. No mask dataset is on disk.** Nothing below is a
dataset result. Every number comes from one of two sources:

1. **Synthetic masks with known answers.** A square shifted by `d` has an
   analytic IoU of `(s-d)/(s+d)`; the tests assert that value, not an
   observed one.
2. **Boxes read as rectangular masks.** The box datasets on disk are run
   through the mask code path. This is degenerate on purpose. It exercises the
   whole path end to end, and it gives the one correctness test available
   today: when a mask happens to be a rectangle, the mask metric must return
   what the box metric returns.

The dataset-side number — what the mask code costs on a **real segmentation
shape** — awaits PVSG. See "What PVSG would need" at the end.

---

## The exact command

Run from the repository root. It needs pillow, which only `.venv-local` has,
because reading a VidVRD annotation goes through the canvas scaler in
`latplan/puzzles/puzzle_labeled_objects.py`.

```bash
( ulimit -v 6000000
  MPLCONFIGDIR=/tmp/claude-1000/mpl .venv-local/bin/python tools/planner/m8_mask.py \
      data/video/vidvrd/annotations/train/ILSVRC2015_train_00005005.json \
      --limit 40 --out-dir eval/M8 )
```

Drop the annotation argument to run the synthetic half alone, with no data and
no pillow.

**Expected runtime: under 30 seconds.** Masks are large — 40 frames of 3
objects on the 300x200 canvas is about 7 MB as booleans — so `--limit` is
there to keep the run small. The memory cap is not a formality; an unbounded
run crashed this workstation on 2026-08-28.

Outputs, both under the gitignored `eval/`:

- `eval/M8/m8_mask.json`
- `eval/M8/m8_mask.svg`

Validate the figure with:

```bash
python3 -c "import xml.dom.minidom,sys; xml.dom.minidom.parse(sys.argv[1])" eval/M8/m8_mask.svg
```

Tests:

```bash
.venv-local/bin/python -m unittest discover -s tools/planner/tests
```

---

## What each outcome means, decided in advance

### H8a — agreement. Does the mask metric measure what the box metric measures?

Both sides are rasterised and scored, and `bbox_iou` is **called** rather than
copied, so the comparison is against the real box metric.

| result | conclusion |
|---|---|
| **integer boxes: max gap 0** | The mask metric is the box metric on rectangles. Every mask number below can be placed beside a box number. **This is a pass or fail, not a scale**: any non-zero gap on integer boxes is a bug and everything after it is void. |
| **integer boxes: max gap above 0** | The metric is measuring something else. Stop; no mask result means anything until it is found. |
| **raw canvas boxes: max gap below 0.05** | The conversion costs less than the rasteriser's rounding. Expected, and it is a property of `boxes_to_masks`, not of the metric. |
| **raw canvas boxes: max gap above 0.05** | The canvas is too coarse to hold the dataset boxes and a mask pipeline on this canvas would blur small objects. That would be a real finding about the 300x200 canvas rather than about masks. |

### H8b — the floor. What does the mask representation cost before any planner?

Every box result in this project is reported against its own floor
(`oracle.round_trip_error`). A mask number without the same treatment cannot
be compared with any of them. The mask analogue of the box's one-hot
coordinate code is a **grid occupancy code**: one bit per cell of a
`bins_y` by `bins_x` grid, per object.

The bit count is known in advance and is not an outcome: at the resolution the
real decoder uses, 60 by 40, a box costs **200 bits** per object and a grid
mask costs **2400**, a factor of 12. The outcome is what those bits buy.

| result | conclusion |
|---|---|
| **mask floor IoU at or above the box floor IoU** | The grid code holds an object at least as well as the coordinate code does. Masks are then affordable in accuracy and expensive in width, and the width is the thing to worry about for a `U x P` latent. The next question is a compression, not a metric. |
| **mask floor IoU below the box floor IoU** | The grid code is the weaker representation at this resolution. A mask dataset would need a finer grid or a different code before its numbers could sit beside the box results, and the finer grid costs bits quadratically. |
| **`vanished` above 0** | The code **deletes** objects smaller than half a cell. That is the failure `oracle._code_width` fixed on the box side, where an object in the top bin decoded as absent. Any dataset with small objects needs a finer grid before its floor means anything, and the count must be reported next to the floor rather than folded into it. |
| **contour F much lower than IoU** | The decoded outline is a staircase at the cell pitch. IoU is blind to that and the contour measure is not, which is the reason both are reported. |

One asymmetry is stated rather than hidden. The box code decodes to the bin's
**left edge**, because that is what a trained decoder emits
(`oracle.dequantise`, where the choice moved the measured floor by a factor of
4.1). That is a deliberate half-bin bias. The grid code decodes by majority
and has none. So the box row is the floor of a representation a model actually
learns, and the mask row is the floor of a representation no model has learned
yet. The run therefore also reports the grid floor at an "any coverage"
threshold, to show how much the floor depends on that one convention.

### H8c — the two metrics. Is one of them enough?

A demonstration with a known answer, not a measurement. Two predictions each
lose exactly 80 pixels of a 400-pixel square: one loses a thin strip from each
end, the other one thick slab from one end.

| result | conclusion |
|---|---|
| **equal IoU, contour F clearly apart** | IoU is blind to where the error sits, so a single mask number would hide a whole class of failure. Report both. |
| **contour F also equal** | The second metric adds nothing at this tolerance and the extra column should be dropped. |

The converse blindness is asserted too: a mask eroded by one pixel scores a
perfect contour F at `tol=2` while its IoU is below 1. **Neither metric is the
answer alone**, and they fail in different directions.

---

## Measured, 2026-09-05

Written after the run. Everything above this heading was written before it.
Confidence marks: **measured** means a command was run and this is its output,
**derived** means arithmetic on measured numbers.

Source: `ILSVRC2015_train_00005005`, first 40 annotated frames, 3 object slots
of which 2 are ever present, so 80 present object-frames. Boxes read as
rectangular masks on the 300x200 canvas.

### H8a — agreement: PASS

| check | result | mark |
|---|---|---|
| integer boxes, largest gap between `mask_iou` and `bbox_iou` over 80 object-frames | **0.000e+00** | measured |
| raw canvas boxes, same gap | **0.000e+00** | measured |
| synthetic non-integer boxes (test suite) | gap **0.0026**, inside the 0.05 bound | measured |

The pre-registered criterion was exact zero on integer boxes, and it is exact
zero. **The mask metric is the box metric on rectangles**, so every mask
number below can be placed beside a box number.

One thing turned out differently from the pre-registration and is worth
recording rather than smoothing over: the two rows are identical because the
canvas scaler in `puzzle_labeled_objects` already **rounds every box to a
whole pixel**. On this pipeline the conversion is therefore lossless, and the
between-pixel case does not arise on real data at all. It is exercised only by
`test_non_integer_boxes_agree_only_to_rasterisation_accuracy`.

### H8b — the floor

Per present object-frame, at the resolution the real decoder uses:

| representation | bits/object | IoU | Dice | contour F | vanished | mark |
|---|---|---|---|---|---|---|
| box one-hot 60x40, left-edge decode | 200 | **0.6864** | 0.7981 | **0.7233** | 0 | measured |
| mask grid 60x40, majority | 2400 | **0.8033** | 0.8868 | **0.9880** | 0 | measured |
| mask grid 60x40, any coverage | 2400 | **0.7281** | 0.8361 | **0.6220** | 0 | measured |

The Dice column is the IoU column restated, `2J/(1+J)`, and it ranks the three
rows in the same order. It is there for readers of the segmentation
literature, not as evidence.

**The pre-registered branch that fired: the mask floor IoU is above the box
floor IoU**, by 0.117 absolute and 17% relative (derived). The reading decided
in advance therefore stands: at this resolution the grid code holds an object
at least as well as the coordinate code does, so **masks are affordable in
accuracy and expensive in width** — 12 times the bits per object (derived,
2400/200). For a `U x P` binary latent the width is the thing to worry about,
and the next question is a compression rather than a metric.

`vanished` is 0 in all three rows (measured), so nothing was deleted on this
clip. That is not general: this clip's objects are large, and a dataset with
small objects has to be checked again before its floor means anything.

**Not observed:** the pre-registered "contour F much lower than IoU" case. The
grid code scores contour F 0.9880 against IoU 0.8033, so the staircase left by
5-pixel cells sits almost entirely inside the 2-pixel tolerance. The
prediction was wrong in a way that favours the mask code.

**The decode convention matters as much as it did on the box side.** Moving
the grid threshold from majority to any-coverage costs 0.075 of IoU and 0.366
of contour F (derived). That is the same class of effect as the box code's
left-edge versus bin-centre choice, which moved the measured floor by a factor
of 4.1 (`oracle.dequantise`). Any real mask decoder has to have this
convention pinned before its floor is quoted.

### H8c — the two metrics: both are needed

| prediction | IoU | contour F | mark |
|---|---|---|---|
| thin strip lost at each end | **0.800** | **1.000** | measured |
| one thick slab lost at one end | **0.800** | **0.750** | measured |

Identical IoU, contour F 0.25 apart. No function of IoU can return two values
for one input, so contour F carries information IoU does not.

The saturation case, from the test suite: the same 2-pixel displacement scores
IoU **0.333** on a 4-pixel-wide object and **0.905** on a 40-pixel-wide one,
while contour F at `tol=2` calls both **1.000** (measured).

### What these numbers are not

- **Not a dataset result.** No mask dataset is on disk. The masks here are
  rectangles derived from boxes, which is the degenerate case, and a rectangle
  is friendlier to an axis-aligned grid code than a real segmentation shape
  will be. The mask floor above is therefore an **upper bound** on what the
  grid code will achieve on PVSG (inferred).
- **Not a dataset statistic.** One clip, two objects, forty frames.
- **Not a like-for-like code comparison.** The box row carries the trained
  decoder's deliberate left-edge half-bin bias and the grid row carries none,
  so it is "the floor of a code a model actually learns" against "the floor of
  a code nothing has learned yet".

## How many metrics is this really? Two, not three

### Jaccard and Dice are one metric

Jaccard **is** IoU, so it is already the natural mask measure and no separate
implementation is needed. Dice is a monotone transform of it:

```
Dice = 2J / (1 + J)
```

That transform is strictly increasing, so Jaccard and Dice **rank any set of
predictions identically and can never disagree**. Dice carries no independent
information whatever.

`mask_dice` is provided and `mean_dice` is reported, because the segmentation
literature reports Dice and a reader calibrated to that literature should not
have to convert by hand. But it is deliberately implemented as
`dice_from_iou(mask_iou(...))` rather than as a second formula, so nothing can
mistake the two columns for two opinions, and `test_dice_is_exactly_the_
monotone_transform_of_iou` and `test_dice_and_iou_rank_the_same_predictions_
in_the_same_order` hold that property in place.

**A milestone counting distinct evaluation methods must count this pair
once.** Counting it twice would inflate the count with a restatement.

### The boundary measure is a genuine second metric

`boundary_f` is **not** a monotone transform of IoU, and the demonstration is
in the tests rather than in an argument. Two predictions each lose exactly 80
pixels of a 400-pixel square, so their IoU is identical at 0.800 — measured —
while their contour scores are 1.000 and 0.750. No function of IoU can produce
two different values from one input, so contour F is carrying information IoU
does not have.

It earns its place for two reasons the author named:

1. **IoU is blind to where the error sits.** An even one-pixel erosion and one
   thick missing slab of the same area are the same number under IoU.
2. **IoU saturates on small objects.** The same 2-pixel displacement scores
   IoU 0.333 on a 4-pixel-wide object and 0.905 on a 40-pixel-wide one —
   measured. IoU is reporting object size as much as it is reporting error.
   Contour F at `tol=2` calls both of them 1.000, which is the other half of
   the trade.

So the count is **two** independent mask measures, `mask_iou` and
`boundary_f`, with Dice as a restatement of the first.

## What each metric is blind to

| metric | sees | blind to |
|---|---|---|
| `mask_iou` (= Jaccard = Dice restated) | how much of the object is right | **where** the error sits — an even erosion and one thick missing slab of the same area score identically; and it saturates on small objects, where a 2 px error reads as catastrophic |
| `boundary_f` | whether the outline is in the right place, at any object size | error below its tolerance; how far past the tolerance a wrong outline has gone (3 px out and 300 px out both score nothing); which side of a contour is filled |

Neither is the answer alone. They fail in different directions, which is the
whole reason to carry both.

---

## The conversion, and why both routes are wanted

`masks_to_boxes` returns the tight enclosing box, half-open so `x2 - x1` is the
number of covered columns, and an all-zero box for an empty mask — the padding
convention every loader here uses.

It is not an alternative to the mask metric. It is the cheaper of two routes
and it answers a different question:

- **Convert to boxes** to compare a mask dataset with VidVRD, VidOR and Action
  Genome under `bbox_mse`, `mse_ratio`, `floor_ratio` and the planner, with no
  new code at all. The cost is that the shape is thrown away.
- **Score as masks** to ask whether the shape carried anything the box did
  not. The cost is 12 times the latent width.

---

## What PVSG would need in order to produce the real number

Everything below is missing, and none of it is code:

1. **The dataset on disk.** PVSG is not downloaded. It is **not** hosted on
   Hugging Face; the release lives on `entuedu-my.sharepoint.com`, which is
   now in the sandbox allowlist, so the download is unblocked and simply has
   not been done. Its masks arrive per frame per object, so the loader has to
   place each object in a stable slot across the clip — the same requirement
   `oracle.boxes_from_vidvrd` meets by keying on the annotation's own track
   id, and the same trap it fell into when it keyed on box area instead and
   let two objects swap slots at a crossing.
2. **A `masks_from_pvsg` reader** returning `(n_frames, num_objs, H, W)`
   booleans on the 300x200 canvas, beside the existing `boxes_from_*`
   readers. The rescale must come from `puzzle_labeled_objects` and not be
   copied (SPEC V5), and rescaling a mask means resampling it, which needs a
   choice of interpolation that this module does not make for you.
3. **A grid resolution decided against the dataset.** `vanished` decides it:
   if any real object is smaller than half a cell at 60 by 40, the floor is
   measuring a deletion rather than a blur.

Until all three exist, the honest statement is the one the run prints: no
number here comes from a real mask dataset.
