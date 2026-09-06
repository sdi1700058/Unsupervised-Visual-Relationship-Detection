#!/usr/bin/env python3
"""M7 -- mean average precision on relation triplets, by the field's own rule.

**The gap this closes.** `RELATED_WORK.md` section B holds the published
ImageNet-VidVRD ladder: mAP 8.58 in 2017, 9.52, 16.26, 18.38, 19.77, and 31.33
in 2024, with 43.15 under VrdONE's oracle-trajectory condition. `EVAL.md` §5.3
records that this project has never computed that quantity. M1
(`predicate_probe.py`) computes a *different* one: average precision per
predicate over per-frame rows, macro-averaged over predicates. The field
averages average precision over **videos**, on **temporally localised triplet
instances** matched by trajectory overlap. The two numbers are not
interchangeable, and quoting one beside the ladder would be a false
comparison. This module computes the ladder's own number.

Which variant, and where it came from
-------------------------------------

This is a port of the dataset author's own toolkit, not a reconstruction from
the paper text. Read on 2026-09-05 from:

- https://github.com/xdshang/VidVRD-helper/blob/master/evaluation/visual_relation_detection.py
- https://github.com/xdshang/VidVRD-helper/blob/master/evaluation/common.py
- https://github.com/xdshang/VidVRD-helper/blob/master/dataset/dataset.py

The same toolkit scores VidVRD and VidOR, and both datasets ship the same
annotation schema, so one implementation serves both.

The protocol, exactly:

**Relation detection.** Predictions for a video are ranked by score. A
prediction matches a ground-truth instance when the two triplets are equal as
strings AND ``min(subject vIoU, object vIoU) >= 0.5``. Matching is greedy from
the highest score down, and each ground-truth instance is consumed at most
once, so flooding the output with copies cannot raise the score. Average
precision uses the all-point VOC rule with the precision envelope. **The mean
is taken over videos**, and a video with no ground-truth relation is dropped
rather than scored zero.

**Relation tagging.** Precision@1, @5 and @10 over the distinct predicted
triplets, in score order, with localisation ignored.

**Recall@50 and Recall@100** pool the top-K hits of every video into one
ranked list and divide by the dataset-wide count of ground-truth instances.
They are NOT averaged per video, which is why they sit beside mAP rather than
inside it.

Volumetric IoU is the sum over the shared frames of the box intersection,
divided by the sum of the two whole trajectory volumes minus that
intersection. Boxes use the inclusive-pixel convention, so a width is
``xmax - xmin + 1``. Dropping the ``+ 1`` moves every number, which is why
`test_m7_map.py` pins it.

Two deviations from the source, both of which only touch cases where the
source raises an exception:

- an empty prediction list gives recall 0 rather than an ``IndexError``;
- a ground-truth instance whose subject or object has no box on some frame of
  its interval is dropped rather than crashing the run. Measured on the whole
  VidVRD test split and on 60 VidOR validation videos: **zero** instances are
  dropped, so this changes no number reported here.

The vectorised `viou` is checked against a literal transcription of the
reference loop on 300 random trajectory pairs.

What is scored, and what is not
-------------------------------

mAP scores *predicted* relations. **No export on disk carries model-predicted
relations for either dataset**, so this module supplies three predictors that
need no model and no GPU, and every one of them is given **ground-truth
tubelets**. That places all of them in VrdONE's oracle-trajectory regime
(`EVAL.md` §5.4), so the row to compare against is 43.15, never 31.33.

==============  ==============================================================
`perfect`       the ground truth returned as its own prediction. **Must give
                mAP exactly 1.000.** It is the only check that says the
                harness is not lying, and every other number is void without
                it.
`frequency`     the most frequent triplets of the training split, applied to
                every ordered pair of ground-truth tubelets. The no-vision
                floor: what a model that sees nothing still scores.
`probe`         the M1 ridge probe read off the oracle latent, cut into
                30-frame segments and greedily associated. That segment-then-
                associate design is VidVRD's own (`RELATED_WORK.md` B1),
                minus the detector, because the tubelets are given.
==============  ==============================================================

`probe` fits the same linear map M1 fits, on the same dataset split, so the two
metrics describe one model. `RidgeAccumulator` builds the normal equations one
clip at a time because stacking every VidOR row needs about two gigabytes;
`test_m7_map.py` pins it against `predicate_probe.ridge_probe_multi`.

numpy and the standard library only, so it runs on the cluster's Python 3.6.

    python3 tools/planner/m7_map.py \\
        --annotations 'data/video/vidvrd/annotations/test/*.json' \\
        --predictor frequency \\
        --prior 'data/video/vidvrd/annotations/train/*.json' \\
        --tag vidvrd-frequency --out-dir eval/M7

Reading the result
------------------

======================  =======================================================
`perfect` below 1.000   the harness is broken. Report nothing else.
`frequency` at 0.000    the predictions never localise. That is a bug in the
                        association step, not a floor.
probe at or below       the latent adds nothing this metric can see. It agrees
`frequency`             with M1's measured lift of -0.003 and it is a fact
                        about a purely positional code, not about the harness.
probe above             the latent carries relation information the field's
`frequency`             own metric can see, in the oracle-trajectory regime.
======================  =======================================================

First result, 2026-09-05, on the 0 to 100 scale the published tables use
(`experiments/M7_map_triplets/README.md` holds the full table)::

    condition          videos     mAP
    vidvrd-perfect        200  100.00
    vidvrd-frequency      200   30.33
    vidvrd-probe            6    1.81
    vidor-perfect         200  100.00
    vidor-frequency       200   16.94
    vidor-probe             7    3.92

**Read the floor before the probe.** A predictor that sees nothing scores
30.33 on VidVRD, where the published ladder runs from 8.58 to 31.33. Handing
a system ground-truth tubelets removes most of the localisation problem on
these datasets, because most tracks and most relations last the whole clip.
**No row above belongs on the published ladder**, because every row there
detects its own tubelets. Quote the GAP between conditions, never one value.

The probe sits far below the floor on both datasets, which agrees with M1's
measured lift of -0.003 on the same clips, the same split and the same
3,990 training rows. A purely positional code does not express predicates
such as `chase`, and two metrics now say so.
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from tools.planner.predicate_probe import (          # noqa: E402
    _pair_onehot, relation_labels)

# The rows the thesis is measured against, so a figure never shows a bar
# without the ladder beside it. Source: RELATED_WORK.md section B, itself
# read out of the papers.
PUBLISHED = (
    ("VidVRD 2017", 8.58),
    ("VrdONE 2024", 31.33),
    ("VrdONE oracle", 43.15),
)


# --------------------------------------------------------------- geometry

def bbox_iou(box_a, box_b):
    """Intersection over union of two boxes, inclusive-pixel convention.

    A box of `xmin == xmax` is one pixel wide, not zero, which is why every
    extent carries a `+ 1`. The reference toolkit does the same, and matching
    it matters: dropping the `+ 1` shifts every mAP in this module.
    """
    w_a = box_a[2] - box_a[0] + 1
    h_a = box_a[3] - box_a[1] + 1
    w_b = box_b[2] - box_b[0] + 1
    h_b = box_b[3] - box_b[1] + 1
    over_w = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]) + 1)
    over_h = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]) + 1)
    over = float(over_w * over_h)
    return over / (w_a * h_a + w_b * h_b - over)


def viou(traj_1, duration_1, traj_2, duration_2):
    """Volumetric IoU of two trajectories, each a duration and its boxes.

    A duration is `[fstart, fend)`, half open, and the boxes are listed for
    exactly those frames. The volume of a trajectory is the sum of its box
    areas over its whole duration, so a prediction that spans far more frames
    than the ground truth is penalised even where the boxes agree perfectly.
    That is what makes this a *localisation* measure rather than an overlap
    measure, and it is the reason a whole-clip prediction cannot pass at 0.5.

    Vectorised over frames. The scalar reference loop is transcribed in
    `test_m7_map.py` and the two are compared on random trajectories, because
    this vectorisation is the only place where the port could drift from the
    source it claims to follow.
    """
    if duration_1[0] >= duration_2[1] or duration_1[1] <= duration_2[0]:
        return 0.0

    head_1 = max(0, duration_2[0] - duration_1[0])
    head_2 = max(0, duration_1[0] - duration_2[0])
    n = min(duration_1[1], duration_2[1]) - max(duration_1[0], duration_2[0])
    # The source indexes straight into the lists and raises where a
    # trajectory is shorter than the duration it claims. Clamping is
    # identical wherever the source does not raise.
    n = min(n, len(traj_1) - head_1, len(traj_2) - head_2)
    if n <= 0:
        return 0.0

    t_1 = np.asarray(traj_1, dtype=np.float64)
    t_2 = np.asarray(traj_2, dtype=np.float64)
    a = t_1[head_1:head_1 + n]
    b = t_2[head_2:head_2 + n]

    over_w = np.maximum(0.0, np.minimum(a[:, 2], b[:, 2])
                        - np.maximum(a[:, 0], b[:, 0]) + 1.0)
    over_h = np.maximum(0.0, np.minimum(a[:, 3], b[:, 3])
                        - np.maximum(a[:, 1], b[:, 1]) + 1.0)
    v_over = float((over_w * over_h).sum())

    v_1 = float(((t_1[:, 2] - t_1[:, 0] + 1.0)
                 * (t_1[:, 3] - t_1[:, 1] + 1.0)).sum())
    v_2 = float(((t_2[:, 2] - t_2[:, 0] + 1.0)
                 * (t_2[:, 3] - t_2[:, 1] + 1.0)).sum())
    return v_over / (v_1 + v_2 - v_over)


def voc_ap(rec, prec):
    """Area under the precision-recall curve, all-point VOC rule.

    The precision envelope is taken first, so a later, better precision lifts
    every earlier point. The 11-point variant is NOT used: the toolkit
    defaults to this one, and the two disagree by whole points.
    """
    mrec = np.concatenate(([0.0], np.asarray(rec, dtype=np.float64), [1.0]))
    mpre = np.concatenate(([0.0], np.asarray(prec, dtype=np.float64), [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[i + 1] - mrec[i]) * mpre[i + 1]).sum())


# ------------------------------------------------------ ground truth load

def _boxes_per_frame(doc):
    """Frame index -> {tid: (xmin, ymin, xmax, ymax)}."""
    out = []
    for frame in doc.get("trajectories", []):
        row = {}
        for roi in (frame or []):
            b = roi["bbox"]
            row[roi["tid"]] = (b["xmin"], b["ymin"], b["xmax"], b["ymax"])
        out.append(row)
    return out


def relation_instances(doc):
    """The ground-truth relation instances of one annotation document.

    This is `Dataset.get_relation_insts` from the reference toolkit: one
    instance per entry of `relation_instances`, the triplet given as
    **category names** rather than track ids, the duration half open, and the
    two trajectories sliced out of the per-frame boxes.

    An instance is dropped where a box is missing on some frame of its
    interval. Interpolating one would change the volume the metric divides
    by, and dropping is the only choice that cannot inflate a score.
    """
    category = dict((so["tid"], so["category"])
                    for so in doc.get("subject/objects", []))
    frames = _boxes_per_frame(doc)

    out = []
    for rel in doc.get("relation_instances", []):
        begin = int(rel["begin_fid"])
        end = int(rel["end_fid"])
        s_tid = rel["subject_tid"]
        o_tid = rel["object_tid"]
        if begin < 0 or end > len(frames) or end <= begin:
            continue
        window = frames[begin:end]
        if any(s_tid not in f or o_tid not in f for f in window):
            continue
        out.append({
            "triplet": (category[s_tid], rel["predicate"], category[o_tid]),
            "subject_tid": s_tid,
            "object_tid": o_tid,
            "duration": (begin, end),
            "sub_traj": [f[s_tid] for f in window],
            "obj_traj": [f[o_tid] for f in window],
        })
    return out


def tubelets(doc):
    """One tubelet per track: its category, its span and its boxes.

    The span is the **longest run of consecutive frames** the track is
    annotated on. A track that vanishes and returns would otherwise get a
    trajectory shorter than the duration it claims, and `viou` would compare
    the wrong frames.
    """
    category = dict((so["tid"], so["category"])
                    for so in doc.get("subject/objects", []))
    frames = _boxes_per_frame(doc)

    runs = {}
    current = {}
    for f, row in enumerate(frames):
        for tid in list(current):
            if tid not in row:
                del current[tid]
        for tid in row:
            if tid not in current:
                current[tid] = f
            begin = current[tid]
            best = runs.get(tid)
            if best is None or (f + 1 - begin) > (best[1] - best[0]):
                runs[tid] = (begin, f + 1)

    out = []
    for tid in sorted(runs):
        if tid not in category:
            continue
        begin, end = runs[tid]
        out.append({
            "tid": tid,
            "category": category[tid],
            "duration": (begin, end),
            "traj": [frames[f][tid] for f in range(begin, end)],
        })
    return out


# ------------------------------------------------------ the scoring rules

def eval_detection_scores(gt_relations, pred_relations, viou_threshold=0.5):
    """Precision, recall and the hit score of every prediction, in rank order.

    Greedy from the top score down. A prediction may only claim a
    ground-truth instance no earlier prediction claimed, so duplicates score
    as false positives. A non-hit keeps `-inf`, which is what marks it.
    """
    ranked = sorted(pred_relations, key=lambda r: r["score"], reverse=True)
    gt_detected = np.zeros((len(gt_relations),), dtype=bool)
    hit_scores = np.ones((len(ranked),)) * -np.inf

    for p_idx, pred in enumerate(ranked):
        ov_max = -float("inf")
        k_max = -1
        for g_idx, gt in enumerate(gt_relations):
            if gt_detected[g_idx]:
                continue
            if tuple(pred["triplet"]) != tuple(gt["triplet"]):
                continue
            s_iou = viou(pred["sub_traj"], pred["duration"],
                         gt["sub_traj"], gt["duration"])
            o_iou = viou(pred["obj_traj"], pred["duration"],
                         gt["obj_traj"], gt["duration"])
            ov = min(s_iou, o_iou)
            if ov >= viou_threshold and ov > ov_max:
                ov_max = ov
                k_max = g_idx
        if k_max >= 0:
            hit_scores[p_idx] = pred["score"]
            gt_detected[k_max] = True

    tp = np.isfinite(hit_scores)
    cum_tp = np.cumsum(tp).astype(np.float64)
    cum_fp = np.cumsum(~tp).astype(np.float64)
    tiny = float(np.finfo(np.float32).eps)
    rec = cum_tp / max(len(gt_relations), tiny)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, tiny)
    return prec, rec, hit_scores


def eval_tagging_scores(gt_relations, pred_relations):
    """The same ranking with localisation ignored, over distinct triplets.

    A triplet the model names twice counts once, at its best score, so a
    model cannot lift Precision@10 by repeating one confident guess.
    """
    ranked = sorted(pred_relations, key=lambda r: r["score"], reverse=True)
    gt_triplets = set(tuple(r["triplet"]) for r in gt_relations)

    seen = []
    hit_scores = []
    for r in ranked:
        triplet = tuple(r["triplet"])
        if triplet not in seen:
            seen.append(triplet)
            hit_scores.append(r["score"])
    hit_scores = np.asarray(hit_scores, dtype=np.float64)
    for i, triplet in enumerate(seen):
        if triplet not in gt_triplets:
            hit_scores[i] = -np.inf

    tp = np.isfinite(hit_scores)
    cum_tp = np.cumsum(tp).astype(np.float64)
    cum_fp = np.cumsum(~tp).astype(np.float64)
    tiny = float(np.finfo(np.float32).eps)
    rec = cum_tp / max(len(gt_triplets), tiny)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, tiny)
    return prec, rec, hit_scores


def evaluate_stream(items, viou_threshold=0.5, det_nreturns=(50, 100),
                    tag_nreturns=(1, 5, 10)):
    """The protocol over an iterable of `(video_id, ground truth, prediction)`.

    Streaming rather than dictionary-at-a-time because VidOR's validation
    annotations are 278 MB on disk and holding every trajectory at once is
    the one thing that would put this over the memory cap.
    """
    video_ap = {}
    tot_scores = dict((n, []) for n in det_nreturns)
    tot_tp = dict((n, []) for n in det_nreturns)
    prec_at_n = dict((n, []) for n in tag_nreturns)
    tot_gt = 0

    for vid, gt, pred in items:
        if not gt:
            # A video with nothing to find is dropped, not scored zero. The
            # reference does this, and averaging zeros over empty videos
            # would make mAP depend on how the split was filtered.
            continue
        tot_gt += len(gt)
        det_prec, det_rec, det_scores = eval_detection_scores(
            gt, pred, viou_threshold)
        video_ap[vid] = voc_ap(det_rec, det_prec)
        tp = np.isfinite(det_scores)
        for n in det_nreturns:
            cut = min(n, det_scores.size)
            tot_scores[n].append(det_scores[:cut])
            tot_tp[n].append(tp[:cut])
        tag_prec, _, _ = eval_tagging_scores(gt, pred)
        for n in tag_nreturns:
            cut = min(n, tag_prec.size)
            prec_at_n[n].append(float(tag_prec[cut - 1]) if cut > 0 else 0.0)

    recall = {}
    for n in det_nreturns:
        parts = [p for p in tot_tp[n] if p.size]
        if not parts or tot_gt == 0:
            recall[n] = 0.0
            continue
        scores = np.concatenate([s for s in tot_scores[n] if s.size])
        tps = np.concatenate(parts)
        tps = tps[np.argsort(scores)[::-1]]
        recall[n] = float(np.cumsum(tps)[-1] / float(tot_gt))

    precision = dict((n, float(np.mean(prec_at_n[n])) if prec_at_n[n] else 0.0)
                     for n in tag_nreturns)

    return {
        "mean_ap": float(np.mean(list(video_ap.values()))) if video_ap else 0.0,
        "recall": recall,
        "precision": precision,
        "n_videos": len(video_ap),
        "n_gt_relations": tot_gt,
        "video_ap": video_ap,
        "viou_threshold": viou_threshold,
    }


def evaluate(groundtruth, prediction, viou_threshold=0.5,
             det_nreturns=(50, 100), tag_nreturns=(1, 5, 10)):
    """`evaluate_stream` over two dictionaries, keyed by video id."""
    items = ((vid, groundtruth[vid], prediction.get(vid, []))
             for vid in sorted(groundtruth))
    return evaluate_stream(items, viou_threshold, det_nreturns, tag_nreturns)


# ----------------------------------------------------------- the predictors

def perfect_predictions(gt_relations):
    """The ground truth returned as its own prediction, every score 1.0."""
    out = []
    for gt in gt_relations:
        pred = dict(gt)
        pred["score"] = 1.0
        out.append(pred)
    return out


class TripletPrior(object):
    """How often each `(subject, predicate, object)` appears in a split.

    The no-vision predictor. It answers "what relation usually holds between
    a dog and a frisbee", which needs no pixels at all, so whatever it scores
    is the part of the metric that carries no information about the model.
    """

    def __init__(self):
        self.by_pair = defaultdict(lambda: defaultdict(int))
        self.overall = defaultdict(int)
        self.n_videos = 0

    def count(self, doc):
        category = dict((so["tid"], so["category"])
                        for so in doc.get("subject/objects", []))
        self.n_videos += 1
        for rel in doc.get("relation_instances", []):
            s = category.get(rel["subject_tid"])
            o = category.get(rel["object_tid"])
            if s is None or o is None:
                continue
            self.by_pair[(s, o)][rel["predicate"]] += 1
            self.overall[rel["predicate"]] += 1

    def top(self, subject, obj, k):
        """The k most frequent predicates for this category pair.

        An unseen pair falls back to the dataset-wide predicate frequency,
        because returning nothing would give the pair a free zero rather than
        the honest guess a frequency model would make.
        """
        counts = self.by_pair.get((subject, obj)) or self.overall
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]


def frequency_predictions(doc, prior, top_k=10, max_per_video=200):
    """The frequency prior applied to every ordered pair of ground-truth tubelets.

    The duration of a prediction is the span the two tubelets share, so the
    prediction is as well localised as the tubelets allow. Ordering is fully
    deterministic, including the ties, which are many: a frequency model
    gives whole groups of triplets the same score, and an arbitrary order
    among them would make the mAP irreproducible.
    """
    tubs = tubelets(doc)
    out = []
    for sub in tubs:
        for obj in tubs:
            if sub["tid"] == obj["tid"]:
                continue
            begin = max(sub["duration"][0], obj["duration"][0])
            end = min(sub["duration"][1], obj["duration"][1])
            if end <= begin:
                continue
            s_off = begin - sub["duration"][0]
            o_off = begin - obj["duration"][0]
            n = end - begin
            for predicate, count in prior.top(sub["category"],
                                              obj["category"], top_k):
                out.append({
                    "triplet": (sub["category"], predicate, obj["category"]),
                    "duration": (begin, end),
                    "sub_traj": sub["traj"][s_off:s_off + n],
                    "obj_traj": obj["traj"][o_off:o_off + n],
                    "score": float(count),
                })
    out.sort(key=lambda r: (-r["score"], r["triplet"], r["duration"]))
    return out[:max_per_video]


def associate_segments(kept, n_frames, segment):
    """Merge runs of adjacent segments carrying the same triplet.

    This is VidVRD's own relational association (`RELATED_WORK.md` B1) with
    the detector removed: predict on short segments, then stitch neighbours
    that agree. Without it every prediction lasts one segment and a relation
    spanning ten of them can never be localised well enough to match.

    `kept` is `(segment index, key, score)`. The merged score is the mean of
    the segments it came from, so a long confident run does not outrank a
    short one merely by being long.
    """
    by_key = defaultdict(list)
    for seg, key, score in kept:
        by_key[key].append((seg, score))

    out = []
    for key in sorted(by_key, key=lambda k: str(k)):
        runs = sorted(by_key[key])
        start = None
        prev = None
        scores = []
        for seg, score in runs:
            if prev is not None and seg != prev + 1:
                out.append((key, start * segment,
                            min(n_frames, (prev + 1) * segment),
                            float(np.mean(scores))))
                start, scores = None, []
            if start is None:
                start = seg
            scores.append(score)
            prev = seg
        if start is not None:
            out.append((key, start * segment,
                        min(n_frames, (prev + 1) * segment),
                        float(np.mean(scores))))
    out.sort(key=lambda r: (r[1], str(r[0])))
    return out


class RidgeAccumulator(object):
    """The normal equations of M1's ridge probe, built one clip at a time.

    `predicate_probe.ridge_probe_multi` stacks every row before solving.
    VidOR's twenty-five exports come to about 390,000 rows of 606 columns,
    which is roughly two gigabytes in double precision, and the fancy-indexed
    training slice doubles it. `X.T @ X` and `X.T @ Y` are both sums over
    rows, so they can be accumulated per clip at a few megabytes.

    The result is the same map, and `test_m7_map.py` pins the two together.
    The intercept column is never penalised, exactly as M1 does it.
    """

    def __init__(self, n_features, n_targets, alpha=1.0):
        self.n_features = n_features
        self.gram = np.zeros((n_features + 1, n_features + 1))
        self.rhs = np.zeros((n_features + 1, n_targets))
        self.alpha = alpha
        self.n_rows = 0

    @staticmethod
    def _with_intercept(x):
        x = np.asarray(x, dtype=np.float64)
        return np.hstack([x, np.ones((len(x), 1))])

    def add(self, x, y):
        xi = self._with_intercept(x)
        self.gram += xi.T.dot(xi)
        self.rhs += xi.T.dot(np.asarray(y, dtype=np.float64))
        self.n_rows += len(xi)

    def solve(self):
        reg = self.alpha * np.eye(self.n_features + 1)
        reg[-1, -1] = 0.0
        return np.linalg.solve(self.gram + reg, self.rhs)

    def predict(self, weights, x):
        return self._with_intercept(x).dot(weights)


# ------------------------------------------------------ the probe condition

def distinct_latents(latents):
    """How many different codes an export actually holds.

    A dead export carries exactly one code, every bit zero, and a mAP
    computed on it measures a broken encoder rather than a method. Four
    exports under `eval/exports/` are in that state, so this count is
    reported beside every probe number instead of being assumed, and a dead
    export is dropped rather than scored.
    """
    z = np.ascontiguousarray(np.asarray(latents))
    z = z.reshape(len(z), -1)
    if z.shape[1] == 0:
        return 0
    view = z.view(np.dtype((np.void, z.dtype.itemsize * z.shape[1])))
    return int(len(np.unique(view)))


def _probe_features(latents, labels, max_slots):
    """The rows M1 fits on: the latent of the frame, plus which pair it is."""
    z = np.asarray(latents, dtype=np.float64)
    keep = labels.frames < len(z)
    frames = labels.frames[keep]
    pairs = labels.pairs[keep]
    if not len(frames):
        return None
    x = np.hstack([z[frames], _pair_onehot(pairs, max_slots)])
    return frames, pairs, x, labels.Y[keep]


def _relabel(y, own_predicates, vocabulary):
    """Widen a clip's label block to the dataset vocabulary.

    A clip that never shows `chase` must contribute genuine negatives for it,
    or the probe learns the vocabulary of whichever clips it happened to see.
    """
    out = np.zeros((len(y), len(vocabulary)))
    for j, name in enumerate(own_predicates):
        out[:, vocabulary.index(name)] = y[:, j]
    return out


def probe_predictions(clips, segment=30, top_k=10, max_per_video=200,
                      test_frac=0.3, seed=0, alpha=1.0):
    """Fit M1's probe on held-in clips and predict relations on held-out ones.

    Whole clips are held out, with the same rule and the same seed
    `predicate_probe.probe_corpus` uses, so M7's probe condition and M1's
    reported number describe one model on one split. Consecutive frames of a
    video are near duplicates, so a random row split would let a frame's own
    neighbour sit on the other side and every score would be meaningless.

    `clips` is a sequence of `(video_id, latents, document, Labels)`.
    Returns `(predictions by video id, diagnostics)`.
    """
    # A dead export would otherwise contribute rows of identical zeros, which
    # the probe fits happily and which say nothing about any representation.
    codes = [distinct_latents(z) for _, z, _, _ in clips]
    n_dead = sum(1 for n in codes if n <= 1)
    for (vid, _, _, _), n in zip(clips, codes):
        if n <= 1:
            print("skip %s: the export holds %d distinct latent, so it "
                  "measures a broken encoder" % (vid, n))
    clips = [c for c, n in zip(clips, codes) if n > 1]
    codes = [n for n in codes if n > 1]

    if len(clips) < 2:
        raise SystemExit("the probe needs at least two live clips to hold "
                         "one out")

    vocabulary = sorted(set(p for _, _, _, lab in clips
                            for p in lab.predicates))
    if not vocabulary:
        raise SystemExit("no predicate appears in any clip")
    max_slots = max(len(lab.tids) for _, _, _, lab in clips)

    rng = np.random.RandomState(seed)
    order = rng.permutation(len(clips))
    n_test = max(1, int(round(len(clips) * test_frac)))
    test_clips = set(order[:n_test].tolist())

    # Two passes over the clips, building each feature block only when it is
    # needed and dropping it again. One VidOR clip is about 75 MB of double
    # precision rows, so holding all twenty-five at once is close to two
    # gigabytes and the workstation has been crashed once by exactly that.
    acc = None
    n_train_clips = 0
    for c, (vid, latents, doc, lab) in enumerate(clips):
        if c in test_clips:
            continue
        got = _probe_features(latents, lab, max_slots)
        if got is None:
            continue
        frames, pairs, x, y = got
        if acc is None:
            acc = RidgeAccumulator(x.shape[1], len(vocabulary), alpha=alpha)
        elif x.shape[1] != acc.n_features:
            raise SystemExit("clips disagree on latent width")
        acc.add(x, _relabel(y, lab.predicates, vocabulary))
        n_train_clips += 1
    if acc is None:
        raise SystemExit("no held-in clip had latents covering its frames")
    weights = acc.solve()

    predictions = {}
    n_covered = 0
    n_total = 0
    for c, (vid, latents, doc, lab) in enumerate(clips):
        if c not in test_clips:
            continue
        got = _probe_features(latents, lab, max_slots)
        if got is None:
            continue
        frames, pairs, x, _ = got
        if x.shape[1] != acc.n_features:
            raise SystemExit("clips disagree on latent width")
        predictions[vid] = _segment_and_associate(
            doc, lab, frames, pairs, acc.predict(weights, x), vocabulary,
            segment, top_k, max_per_video)
        slots = set(lab.tids)
        for inst in relation_instances(doc):
            n_total += 1
            if (inst["subject_tid"] in slots and inst["object_tid"] in slots
                    and inst["triplet"][1] in vocabulary):
                n_covered += 1

    diagnostics = {
        "n_clips": len(clips),
        "n_train_clips": n_train_clips,
        "n_test_clips": len(predictions),
        "n_train_rows": acc.n_rows,
        "n_predicates": len(vocabulary),
        "n_slots": max_slots,
        "segment": segment,
        # Read this before reading the mAP. One distinct latent means a dead
        # export, and a number computed on one is about the encoder that
        # wrote it rather than about the metric or the method.
        "n_dead_exports": n_dead,
        "distinct_latents_min": min(codes),
        "distinct_latents_median": sorted(codes)[len(codes) // 2],
        # What share of the ground truth the probe could possibly find: it
        # only ever sees the largest few tracks, so relations on the rest are
        # unreachable by construction and the mAP has to be read knowing it.
        "reachable_fraction": (float(n_covered) / n_total) if n_total else 0.0,
        "n_gt_in_test": n_total,
    }
    return predictions, diagnostics


def _segment_and_associate(doc, lab, frames, pairs, scores, vocabulary,
                           segment, top_k, max_per_video):
    """Per-frame probe scores to localised relation instances."""
    tubs = dict((t["tid"], t) for t in tubelets(doc))
    slot_tid = list(lab.tids)
    pair_list = sorted(set(tuple(p) for p in pairs))
    pair_index = dict((p, i) for i, p in enumerate(pair_list))

    n_frames = int(frames.max()) + 1
    n_seg = int(math.ceil(n_frames / float(segment)))
    total = np.zeros((n_seg, len(pair_list), len(vocabulary)))
    count = np.zeros((n_seg, len(pair_list), 1))
    for r in range(len(frames)):
        seg = int(frames[r]) // segment
        total[seg, pair_index[tuple(pairs[r])]] += scores[r]
        count[seg, pair_index[tuple(pairs[r])]] += 1.0
    mean = total / np.maximum(count, 1.0)

    kept = []
    flat = mean.reshape(n_seg, -1)
    width = len(vocabulary)
    for seg in range(n_seg):
        if not count[seg].any():
            continue
        k = min(top_k, flat.shape[1])
        best = np.argpartition(-flat[seg], k - 1)[:k]
        for cell in sorted(best, key=lambda c: (-flat[seg][c], c)):
            kept.append((seg, (int(cell // width), int(cell % width)),
                         float(flat[seg][cell])))

    out = []
    for key, begin, end, score in associate_segments(kept, n_frames, segment):
        pair, predicate = key
        s_tid = slot_tid[pair_list[pair][0]]
        o_tid = slot_tid[pair_list[pair][1]]
        sub = tubs.get(s_tid)
        obj = tubs.get(o_tid)
        if sub is None or obj is None:
            continue
        lo = max(begin, sub["duration"][0], obj["duration"][0])
        hi = min(end, sub["duration"][1], obj["duration"][1])
        if hi <= lo:
            continue
        out.append({
            "triplet": (sub["category"], vocabulary[predicate],
                        obj["category"]),
            "duration": (lo, hi),
            "sub_traj": sub["traj"][lo - sub["duration"][0]:
                                    hi - sub["duration"][0]],
            "obj_traj": obj["traj"][lo - obj["duration"][0]:
                                    hi - obj["duration"][0]],
            "score": score,
        })
    out.sort(key=lambda r: (-r["score"], r["triplet"], r["duration"]))
    return out[:max_per_video]


# --------------------------------------------------------------- figures

def _esc(text):
    """Escape for SVG text. A raw angle bracket is invalid XML."""
    return (str(text).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def result_svg(result, path, width=760):
    """One panel: every headline number of one condition, on a 0 to 100 scale.

    Percent rather than a fraction, because that is the scale every row of
    the published ladder uses and a figure that needs a unit conversion
    before it can be compared is a figure that invites the wrong comparison.
    """
    bars = [("mAP", result["mean_ap"] * 100.0),
            ("R@50", result["recall"].get(50, result["recall"].get("50", 0.0))
             * 100.0),
            ("R@100", result["recall"].get(100, result["recall"].get("100", 0.0))
             * 100.0),
            ("P@1", result["precision"].get(1, result["precision"].get("1", 0.0))
             * 100.0),
            ("P@5", result["precision"].get(5, result["precision"].get("5", 0.0))
             * 100.0),
            ("P@10", result["precision"].get(
                10, result["precision"].get("10", 0.0)) * 100.0)]

    left = 90
    span = float(width - left - 70)
    row_h = 24
    height = 62 + len(bars) * row_h

    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'font-family="sans-serif">' % (width, height)]
    out.append('<text x="8" y="18" font-size="13" fill="#222">M7 %s</text>'
               % _esc(result.get("tag", "")))
    out.append('<text x="8" y="34" font-size="11" fill="#666">%s, %d videos, '
               '%d ground-truth relations, vIoU %.2f. Scale 0 to 100.</text>'
               % (_esc(result.get("dataset", "")), result["n_videos"],
                  result["n_gt_relations"], result["viou_threshold"]))
    for i, (name, value) in enumerate(bars):
        y = 50 + i * row_h
        out.append('<text x="8" y="%d" font-size="11" fill="#333">%s</text>'
                   % (y + 13, _esc(name)))
        out.append('<rect x="%d" y="%d" width="%.1f" height="%d" '
                   'fill="#eef0f4"/>' % (left, y, span, row_h - 6))
        out.append('<rect x="%d" y="%d" width="%.1f" height="%d" '
                   'fill="#1f6feb" opacity="0.85"/>'
                   % (left, y, span * min(value, 100.0) / 100.0, row_h - 6))
        out.append('<text x="%.1f" y="%d" font-size="10" fill="#444">'
                   '%.2f</text>' % (left + span + 6, y + 13, value))
    out.append('</svg>')
    with open(path, "w") as f:
        f.write("".join(out))


def comparison_svg(rows, path, width=820):
    """Every condition on one axis, with the published ladder marked.

    The reference rules are the point of the figure. A bar of this project's
    own next to nothing says how well it did; a bar next to 8.58 and 43.15
    says whether the number is worth reporting at all.
    """
    rows = sorted(rows, key=lambda r: (r.get("dataset", ""), r.get("tag", "")))
    left = 190
    span = float(width - left - 60)
    row_h = 26
    height = 76 + len(rows) * row_h

    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'font-family="sans-serif">' % (width, height)]
    out.append('<text x="8" y="18" font-size="13" fill="#222">M7 relation '
               'detection mAP, VidVRD protocol, 0 to 100</text>')
    out.append('<text x="8" y="34" font-size="11" fill="#666">Dashed rules '
               'are published ImageNet-VidVRD rows (RELATED_WORK.md section '
               'B). Every bar here is given ground-truth tubelets.</text>')

    top = 46
    bottom = top + len(rows) * row_h
    for name, value in PUBLISHED:
        x = left + span * value / 100.0
        out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#b04a3a"'
                   ' stroke-width="1" stroke-dasharray="3,3"/>'
                   % (x, top, x, bottom))
        out.append('<text x="%.1f" y="%d" font-size="9" fill="#b04a3a">%s '
                   '%.2f</text>' % (x + 2, bottom + 12, _esc(name), value))

    for i, row in enumerate(rows):
        y = top + i * row_h
        value = float(row.get("mean_ap", 0.0)) * 100.0
        label = "%s" % row.get("tag", "")
        if len(label) > 30:
            label = label[:29] + "..."
        out.append('<text x="8" y="%d" font-size="11" fill="#333">%s</text>'
                   % (y + 14, _esc(label)))
        out.append('<rect x="%d" y="%d" width="%.1f" height="%d" '
                   'fill="#eef0f4"/>' % (left, y, span, row_h - 8))
        out.append('<rect x="%d" y="%d" width="%.1f" height="%d" '
                   'fill="#1f6feb" opacity="0.85"/>'
                   % (left, y, span * min(value, 100.0) / 100.0, row_h - 8))
        out.append('<text x="%.1f" y="%d" font-size="10" fill="#444">%.2f '
                   '(%d videos)</text>'
                   % (left + span + 4, y + 13, value, row.get("n_videos", 0)))
    out.append('</svg>')
    with open(path, "w") as f:
        f.write("".join(out))


# ------------------------------------------------------------ the command

def _expand(patterns, limit=0):
    paths = []
    for pattern in patterns or []:
        found = sorted(glob.glob(pattern))
        if not found and os.path.exists(pattern):
            found = [pattern]
        paths.extend(found)
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique[:limit] if limit else unique


def _video_id(path):
    """The video id an annotation or an export file belongs to.

    VidOR exports are named `<directory>-<video id>.npz`, so the directory
    prefix is stripped. VidVRD ids carry no hyphen, so nothing is stripped
    from them.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    return name.split("-")[-1]


def _load_prior(patterns, limit):
    prior = TripletPrior()
    for path in _expand(patterns, limit):
        with open(path) as f:
            prior.count(json.load(f))
    if not prior.overall:
        raise SystemExit("the frequency prior saw no relation; check --prior")
    return prior


def _write(result, out_dir, tag):
    target = os.path.join(out_dir, tag)
    if not os.path.isdir(target):
        os.makedirs(target)
    payload = dict(result)
    payload["recall"] = dict((str(k), v) for k, v in result["recall"].items())
    payload["precision"] = dict((str(k), v)
                                for k, v in result["precision"].items())
    with open(os.path.join(target, "map.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    result_svg(result, os.path.join(target, "map.svg"))
    return target


def _report(result):
    print("videos scored   %d" % result["n_videos"])
    print("ground truth    %d relation instances" % result["n_gt_relations"])
    print("")
    print("  relation detection mAP   %6.2f" % (result["mean_ap"] * 100.0))
    print("  recall@50                %6.2f" % (result["recall"][50] * 100.0))
    print("  recall@100               %6.2f" % (result["recall"][100] * 100.0))
    print("  tagging precision@1      %6.2f"
          % (result["precision"][1] * 100.0))
    print("  tagging precision@5      %6.2f"
          % (result["precision"][5] * 100.0))
    print("  tagging precision@10     %6.2f"
          % (result["precision"][10] * 100.0))


def _run_compare(args):
    rows = []
    for path in _expand(args.compare):
        with open(path) as f:
            row = json.load(f)
        rows.append({"tag": row.get("tag", os.path.basename(
            os.path.dirname(path))),
            "dataset": row.get("dataset", ""),
            "mean_ap": row.get("mean_ap", 0.0),
            "n_videos": row.get("n_videos", 0)})
    if not rows:
        raise SystemExit("--compare matched no map.json")
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    target = os.path.join(args.out_dir, "m7_map.svg")
    comparison_svg(rows, target)
    print("%-26s %8s %8s" % ("condition", "mAP", "videos"))
    for row in sorted(rows, key=lambda r: (r["dataset"], r["tag"])):
        print("%-26s %8.2f %8d"
              % (row["tag"], row["mean_ap"] * 100.0, row["n_videos"]))
    print("\nwrote %s" % target)
    return 0


def _probe_clips(paths, exports, max_objects):
    by_id = dict((_video_id(p), p) for p in paths)
    clips = []
    for export in exports:
        vid = _video_id(export)
        ann = by_id.get(vid)
        if ann is None:
            print("skip %s: no annotation" % os.path.basename(export))
            continue
        latents = np.load(export)["latents"]
        with open(ann) as f:
            doc = json.load(f)
        try:
            labels = relation_labels(ann, num_objs=max_objects)
        except SystemExit as exc:
            print("skip %s: %s" % (vid, exc))
            continue
        clips.append((vid, latents, doc, labels))
    return clips


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations", nargs="+", default=[],
                    help="glob patterns for the annotation json files to "
                         "score against")
    ap.add_argument("--predictor", default="perfect",
                    choices=["perfect", "frequency", "probe"])
    ap.add_argument("--prior", nargs="+", default=[],
                    help="glob patterns for the TRAINING split the frequency "
                         "predictor counts triplets in")
    ap.add_argument("--exports", nargs="+", default=[],
                    help="glob patterns for the latent npz files the probe "
                         "predictor reads")
    ap.add_argument("--viou", type=float, default=0.5,
                    help="the field's threshold. Change it only with a reason,"
                         " and say so beside the number")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-per-video", type=int, default=200)
    ap.add_argument("--segment", type=int, default=30,
                    help="segment length for the probe predictor. 30 is "
                         "VidVRD's own")
    ap.add_argument("--max-objects", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prior-limit", type=int, default=0)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--compare", nargs="+", default=[],
                    help="glob patterns for map.json files to draw on one axis")
    args = ap.parse_args(argv)

    if args.compare:
        if not args.out_dir:
            raise SystemExit("--compare needs --out-dir")
        return _run_compare(args)

    paths = _expand(args.annotations, args.limit)
    if not paths:
        raise SystemExit("--annotations matched nothing")

    diagnostics = {}
    if args.predictor == "probe":
        exports = _expand(args.exports)
        if not exports:
            raise SystemExit("the probe predictor needs --exports")
        # The probe scores only the clips it holds out, so the annotation set
        # is narrowed to those rather than the whole split. `paths` rather
        # than a second expansion, so --limit means the same thing here as it
        # does for every other predictor.
        clips = _probe_clips(paths, exports, args.max_objects)
        predictions, diagnostics = probe_predictions(
            clips, segment=args.segment, top_k=args.top_k,
            max_per_video=args.max_per_video, test_frac=args.test_frac,
            seed=args.seed)
        held = dict((vid, doc) for vid, _, doc, _ in clips
                    if vid in predictions)
        items = [(vid, relation_instances(held[vid]), predictions[vid])
                 for vid in sorted(held)]
        result = evaluate_stream(items, args.viou)
    else:
        prior = None
        if args.predictor == "frequency":
            if not args.prior:
                raise SystemExit("the frequency predictor needs --prior")
            prior = _load_prior(args.prior, args.prior_limit)
            diagnostics = {"prior_videos": prior.n_videos,
                           "prior_predicates": len(prior.overall),
                           "top_k": args.top_k,
                           "max_per_video": args.max_per_video}

        def stream():
            for path in paths:
                with open(path) as f:
                    doc = json.load(f)
                gt = relation_instances(doc)
                if args.predictor == "perfect":
                    pred = perfect_predictions(gt)
                else:
                    pred = frequency_predictions(doc, prior, args.top_k,
                                                 args.max_per_video)
                yield doc.get("video_id", _video_id(path)), gt, pred

        result = evaluate_stream(stream(), args.viou)

    result["tag"] = args.tag or args.predictor
    result["dataset"] = args.dataset
    result["predictor"] = args.predictor
    result["n_annotations"] = len(paths)
    result["diagnostics"] = diagnostics

    print("M7  %s  (%s, %s)" % (result["tag"], args.predictor,
                                args.dataset or "unnamed"))
    # Before the numbers, not after them. The module docstring's rule is
    # "`perfect` below 1.000: the harness is broken, report nothing else", and
    # a warning printed under a full table is read as a footnote to it.
    broken = (args.predictor == "perfect"
              and abs(result["mean_ap"] - 1.0) > 1e-9)
    if broken:
        print("\nWARNING: the ground truth scored against itself is %.6f, "
              "not 1.0. The harness is wrong and no other M7 number stands."
              % result["mean_ap"])
    _report(result)
    if diagnostics:
        print("")
        for key in sorted(diagnostics):
            print("  %-22s %s" % (key, diagnostics[key]))

    if args.out_dir:
        target = _write(result, args.out_dir, result["tag"])
        print("\nwrote %s/map.json and map.svg" % target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
