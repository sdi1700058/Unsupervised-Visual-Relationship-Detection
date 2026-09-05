#!/usr/bin/env python3
"""Tests for M7, mean average precision on relation triplets.

Every number below is known by construction or worked out by hand from the
published rule, never copied from a run of the code under test. A metric
checked against its own output is not checked at all, and a mAP that is not
the field's mAP invites a false comparison with the ladder in
`RELATED_WORK.md` section B.

Two of the tests compare the vectorised `viou` against a slow, literal
transcription of the reference loop, on random trajectories. That is the test
that guards the optimisation, because the optimisation is the only place where
this port could drift from the source it claims to follow.

    .venv-local/bin/python -m unittest discover -s tools/planner/tests
"""

import json
import os
import random
import sys
import tempfile
import unittest
import xml.dom.minidom

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from tools.planner.m7_map import (          # noqa: E402
    RidgeAccumulator, TripletPrior, associate_segments, bbox_iou,
    comparison_svg, distinct_latents, eval_detection_scores,
    eval_tagging_scores, evaluate, frequency_predictions,
    perfect_predictions, probe_predictions, relation_instances,
    result_svg, tubelets, viou, voc_ap)


# ---------------------------------------------------------------- fixtures

def _doc(video_id="TEST_0001", n_frames=10):
    """A two-object clip whose boxes never move, so every volume is exact."""
    return {
        "video_id": video_id,
        "width": 640, "height": 360, "fps": 30, "frame_count": n_frames,
        "subject/objects": [{"tid": 0, "category": "dog"},
                            {"tid": 1, "category": "frisbee"}],
        "relation_instances": [
            {"subject_tid": 0, "object_tid": 1, "predicate": "chase",
             "begin_fid": 0, "end_fid": 5},
            {"subject_tid": 0, "object_tid": 1, "predicate": "taller",
             "begin_fid": 0, "end_fid": 10},
        ],
        "trajectories": [
            [{"tid": 0, "bbox": {"xmin": 0, "ymin": 0, "xmax": 9, "ymax": 9}},
             {"tid": 1, "bbox": {"xmin": 20, "ymin": 20,
                                 "xmax": 29, "ymax": 29}}]
            for _ in range(n_frames)
        ],
    }


def _box_traj(box, n):
    return [tuple(box) for _ in range(n)]


def _reference_viou(traj_1, duration_1, traj_2, duration_2):
    """A literal transcription of the reference loop, kept slow on purpose.

    Source: https://github.com/xdshang/VidVRD-helper/blob/master/evaluation/common.py
    """
    if duration_1[0] >= duration_2[1] or duration_1[1] <= duration_2[0]:
        return 0.
    elif duration_1[0] <= duration_2[0]:
        head_1 = duration_2[0] - duration_1[0]
        head_2 = 0
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    else:
        head_1 = 0
        head_2 = duration_1[0] - duration_2[0]
        if duration_1[1] < duration_2[1]:
            tail_1 = duration_1[1] - duration_1[0]
            tail_2 = duration_1[1] - duration_2[0]
        else:
            tail_1 = duration_2[1] - duration_1[0]
            tail_2 = duration_2[1] - duration_2[0]
    del tail_2
    v_overlap = 0
    for i in range(tail_1 - head_1):
        roi_1 = traj_1[head_1 + i]
        roi_2 = traj_2[head_2 + i]
        left = max(roi_1[0], roi_2[0])
        top = max(roi_1[1], roi_2[1])
        right = min(roi_1[2], roi_2[2])
        bottom = min(roi_1[3], roi_2[3])
        v_overlap += max(0, right - left + 1) * max(0, bottom - top + 1)
    v1 = 0
    for i in range(len(traj_1)):
        v1 += ((traj_1[i][2] - traj_1[i][0] + 1)
               * (traj_1[i][3] - traj_1[i][1] + 1))
    v2 = 0
    for i in range(len(traj_2)):
        v2 += ((traj_2[i][2] - traj_2[i][0] + 1)
               * (traj_2[i][3] - traj_2[i][1] + 1))
    return float(v_overlap) / (v1 + v2 - v_overlap)


# ------------------------------------------------------------- box overlap

class TestBboxIou(unittest.TestCase):

    def test_identical_boxes_score_one(self):
        self.assertAlmostEqual(bbox_iou((0, 0, 9, 9), (0, 0, 9, 9)), 1.0)

    def test_quarter_overlap_by_hand(self):
        # Both boxes are 10 by 10 under the inclusive-pixel convention, and
        # they share a 5 by 5 corner: 25 / (100 + 100 - 25) = 1/7.
        self.assertAlmostEqual(bbox_iou((0, 0, 9, 9), (5, 5, 14, 14)),
                               25.0 / 175.0)

    def test_disjoint_boxes_score_zero(self):
        self.assertAlmostEqual(bbox_iou((0, 0, 9, 9), (20, 20, 29, 29)), 0.0)

    def test_touching_boxes_share_one_pixel_column(self):
        # The +1 convention means xmax=9 and xmin=9 overlap by one pixel.
        # This is the detail that shifts every number if it is dropped.
        self.assertGreater(bbox_iou((0, 0, 9, 9), (9, 0, 18, 9)), 0.0)


class TestViou(unittest.TestCase):

    def test_identical_trajectories_score_one(self):
        t = _box_traj((0, 0, 9, 9), 4)
        self.assertAlmostEqual(viou(t, (0, 4), t, (0, 4)), 1.0)

    def test_disjoint_durations_score_zero(self):
        t = _box_traj((0, 0, 9, 9), 4)
        self.assertAlmostEqual(viou(t, (0, 4), t, (4, 8)), 0.0)

    def test_half_overlapping_durations_by_hand(self):
        # Each trajectory is 4 frames of a 10x10 box, so each volume is 400.
        # The durations share 2 frames, so the shared volume is 200 and the
        # answer is 200 / (400 + 400 - 200) = 1/3.
        t = _box_traj((0, 0, 9, 9), 4)
        self.assertAlmostEqual(viou(t, (0, 4), t, (2, 6)), 1.0 / 3.0)

    def test_matches_the_reference_loop_on_random_trajectories(self):
        rng = random.Random(20260905)
        for _ in range(300):
            b1 = rng.randrange(0, 8)
            b2 = rng.randrange(0, 8)
            n1 = rng.randrange(1, 9)
            n2 = rng.randrange(1, 9)
            t1 = [self._box(rng) for _ in range(n1)]
            t2 = [self._box(rng) for _ in range(n2)]
            d1 = (b1, b1 + n1)
            d2 = (b2, b2 + n2)
            self.assertAlmostEqual(viou(t1, d1, t2, d2),
                                   _reference_viou(t1, d1, t2, d2), places=10)

    @staticmethod
    def _box(rng):
        x = rng.randrange(0, 40)
        y = rng.randrange(0, 40)
        return (x, y, x + rng.randrange(1, 20), y + rng.randrange(1, 20))


# --------------------------------------------------------------- VOC AP

class TestVocAp(unittest.TestCase):

    def test_a_perfect_ranking_scores_one(self):
        # Two ground truths, two hits: recall climbs to 1 at precision 1.
        self.assertAlmostEqual(voc_ap([0.5, 1.0], [1.0, 1.0]), 1.0)

    def test_a_miss_before_a_hit_by_hand(self):
        # Two ground truths, one false positive then one hit. The precision
        # envelope is 0.5 up to recall 0.5 and 0 after it, so the area is
        # 0.5 * 0.5 = 0.25.
        self.assertAlmostEqual(voc_ap([0.0, 0.5], [0.0, 0.5]), 0.25)

    def test_no_prediction_scores_zero(self):
        self.assertAlmostEqual(voc_ap([], []), 0.0)


# ------------------------------------------------------- ground truth load

class TestRelationInstances(unittest.TestCase):

    def test_triplet_uses_category_names_not_track_ids(self):
        insts = relation_instances(_doc())
        self.assertEqual(insts[0]["triplet"], ("dog", "chase", "frisbee"))

    def test_end_fid_is_exclusive(self):
        insts = relation_instances(_doc())
        chase = [i for i in insts if i["triplet"][1] == "chase"][0]
        self.assertEqual(chase["duration"], (0, 5))
        self.assertEqual(len(chase["sub_traj"]), 5)
        self.assertEqual(len(chase["obj_traj"]), 5)

    def test_an_instance_with_a_missing_box_is_dropped_not_guessed(self):
        # A relation cannot be scored where its subject has no box, and
        # inventing one would change the volume the metric divides by.
        doc = _doc()
        doc["trajectories"][3] = [t for t in doc["trajectories"][3]
                                  if t["tid"] != 0]
        insts = relation_instances(doc)
        self.assertEqual([i["triplet"][1] for i in insts], [])

    def test_tubelets_report_category_and_span(self):
        tubs = {t["tid"]: t for t in tubelets(_doc())}
        self.assertEqual(tubs[0]["category"], "dog")
        self.assertEqual(tubs[0]["duration"], (0, 10))
        self.assertEqual(len(tubs[0]["traj"]), 10)


# ------------------------------------------------------- detection scoring

def _pred(triplet, duration, sub, obj, score):
    n = duration[1] - duration[0]
    return {"triplet": triplet, "duration": duration, "score": score,
            "sub_traj": _box_traj(sub, n), "obj_traj": _box_traj(obj, n)}


class TestDetectionScores(unittest.TestCase):

    def setUp(self):
        self.gt = relation_instances(_doc())
        self.chase = [g for g in self.gt if g["triplet"][1] == "chase"][0]

    def test_a_perfect_prediction_is_a_hit(self):
        p = _pred(("dog", "chase", "frisbee"), (0, 5),
                  (0, 0, 9, 9), (20, 20, 29, 29), 1.0)
        prec, rec, hits = eval_detection_scores([self.chase], [p], 0.5)
        self.assertTrue(np.isfinite(hits[0]))
        self.assertAlmostEqual(prec[0], 1.0)
        self.assertAlmostEqual(rec[0], 1.0)

    def test_the_wrong_predicate_never_matches_however_good_the_boxes(self):
        p = _pred(("dog", "run", "frisbee"), (0, 5),
                  (0, 0, 9, 9), (20, 20, 29, 29), 1.0)
        _, _, hits = eval_detection_scores([self.chase], [p], 0.5)
        self.assertFalse(np.isfinite(hits[0]))

    def test_the_wrong_object_category_never_matches(self):
        p = _pred(("dog", "chase", "ball"), (0, 5),
                  (0, 0, 9, 9), (20, 20, 29, 29), 1.0)
        _, _, hits = eval_detection_scores([self.chase], [p], 0.5)
        self.assertFalse(np.isfinite(hits[0]))

    def test_overlap_below_the_threshold_is_not_a_hit(self):
        # The subject box is shifted so its vIoU is 1/7, below 0.5, while the
        # object box is exact. The rule takes the MINIMUM of the two.
        p = _pred(("dog", "chase", "frisbee"), (0, 5),
                  (5, 5, 14, 14), (20, 20, 29, 29), 1.0)
        _, _, hits = eval_detection_scores([self.chase], [p], 0.5)
        self.assertFalse(np.isfinite(hits[0]))

    def test_one_ground_truth_absorbs_one_prediction_only(self):
        # Two identical predictions against one ground truth: the first is a
        # true positive and the second is a false positive, never a second
        # hit. Without this the metric rewards flooding the output.
        p = _pred(("dog", "chase", "frisbee"), (0, 5),
                  (0, 0, 9, 9), (20, 20, 29, 29), 0.9)
        q = _pred(("dog", "chase", "frisbee"), (0, 5),
                  (0, 0, 9, 9), (20, 20, 29, 29), 0.8)
        _, _, hits = eval_detection_scores([self.chase], [p, q], 0.5)
        self.assertTrue(np.isfinite(hits[0]))
        self.assertFalse(np.isfinite(hits[1]))

    def test_predictions_are_ranked_by_score_not_by_input_order(self):
        good = _pred(("dog", "chase", "frisbee"), (0, 5),
                     (0, 0, 9, 9), (20, 20, 29, 29), 0.9)
        bad = _pred(("dog", "run", "frisbee"), (0, 5),
                    (0, 0, 9, 9), (20, 20, 29, 29), 0.1)
        _, _, hits = eval_detection_scores([self.chase], [bad, good], 0.5)
        self.assertTrue(np.isfinite(hits[0]))
        self.assertFalse(np.isfinite(hits[1]))


class TestTaggingScores(unittest.TestCase):

    def test_precision_at_one_and_five_by_hand(self):
        gt = relation_instances(_doc())
        chase = [g for g in gt if g["triplet"][1] == "chase"]
        preds = [_pred(("dog", "chase", "frisbee"), (0, 5),
                       (0, 0, 9, 9), (20, 20, 29, 29), 0.9),
                 _pred(("dog", "run", "frisbee"), (0, 5),
                       (0, 0, 9, 9), (20, 20, 29, 29), 0.8),
                 _pred(("dog", "chase", "frisbee"), (5, 10),
                       (0, 0, 9, 9), (20, 20, 29, 29), 0.7)]
        prec, _, _ = eval_tagging_scores(chase, preds)
        # The repeated triplet is dropped, so the ranked list is one hit then
        # one miss: precision 1.0 then 0.5.
        self.assertEqual(len(prec), 2)
        self.assertAlmostEqual(prec[0], 1.0)
        self.assertAlmostEqual(prec[1], 0.5)

    def test_tagging_ignores_where_the_boxes_are(self):
        gt = relation_instances(_doc())
        chase = [g for g in gt if g["triplet"][1] == "chase"]
        far = _pred(("dog", "chase", "frisbee"), (0, 5),
                    (500, 500, 509, 509), (600, 600, 609, 609), 0.9)
        prec, _, _ = eval_tagging_scores(chase, [far])
        self.assertAlmostEqual(prec[0], 1.0)


# ------------------------------------------------------------ the protocol

class TestEvaluate(unittest.TestCase):

    def test_the_ground_truth_scored_against_itself_is_exactly_one(self):
        # The single check that says the harness is not lying. Every other
        # number in M7 is void if this one is not 1.000.
        gt = {"a": relation_instances(_doc("a")),
              "b": relation_instances(_doc("b"))}
        pred = dict((v, perfect_predictions(g)) for v, g in gt.items())
        r = evaluate(gt, pred)
        self.assertAlmostEqual(r["mean_ap"], 1.0)
        self.assertAlmostEqual(r["recall"][50], 1.0)
        self.assertAlmostEqual(r["precision"][1], 1.0)

    def test_a_video_with_no_ground_truth_is_skipped_not_scored_zero(self):
        empty = _doc("c")
        empty["relation_instances"] = []
        gt = {"a": relation_instances(_doc("a")),
              "c": relation_instances(empty)}
        pred = {"a": perfect_predictions(gt["a"]), "c": []}
        r = evaluate(gt, pred)
        self.assertEqual(r["n_videos"], 1)
        self.assertAlmostEqual(r["mean_ap"], 1.0)

    def test_an_empty_prediction_scores_zero(self):
        gt = {"a": relation_instances(_doc("a"))}
        r = evaluate(gt, {"a": []})
        self.assertAlmostEqual(r["mean_ap"], 0.0)
        self.assertAlmostEqual(r["recall"][50], 0.0)

    def test_the_mean_is_over_videos_not_over_instances(self):
        # One video is answered perfectly and one is not answered at all.
        # Averaging over videos gives 0.5. Pooling the instances would not,
        # because the two videos hold different numbers of them.
        big = _doc("big")
        big["relation_instances"] = big["relation_instances"] * 1
        big["relation_instances"] = [
            {"subject_tid": 0, "object_tid": 1, "predicate": p,
             "begin_fid": 0, "end_fid": 10}
            for p in ("chase", "taller", "away", "front")]
        gt = {"a": relation_instances(_doc("a")),
              "big": relation_instances(big)}
        pred = {"a": perfect_predictions(gt["a"]), "big": []}
        r = evaluate(gt, pred)
        self.assertAlmostEqual(r["mean_ap"], 0.5)

    def test_per_video_average_precision_is_reported(self):
        gt = {"a": relation_instances(_doc("a"))}
        r = evaluate(gt, {"a": perfect_predictions(gt["a"])})
        self.assertIn("a", r["video_ap"])
        self.assertAlmostEqual(r["video_ap"]["a"], 1.0)


# --------------------------------------------------------- the predictors

class TestFrequencyPredictor(unittest.TestCase):

    def test_the_prior_counts_triplets_from_the_training_split(self):
        prior = TripletPrior()
        prior.count(_doc())
        prior.count(_doc())
        top = prior.top("dog", "frisbee", 5)
        self.assertEqual([t[0] for t in top][:2], ["chase", "taller"])
        self.assertEqual(top[0][1], 2)

    def test_an_unseen_category_pair_falls_back_to_the_global_rate(self):
        prior = TripletPrior()
        prior.count(_doc())
        top = prior.top("zebra", "car", 5)
        self.assertTrue(top)
        self.assertIn(top[0][0], ("chase", "taller"))

    def test_predictions_cover_both_orderings_of_every_pair(self):
        prior = TripletPrior()
        prior.count(_doc())
        preds = frequency_predictions(_doc(), prior, top_k=2,
                                      max_per_video=100)
        subjects = set(p["triplet"][0] for p in preds)
        self.assertEqual(subjects, set(["dog", "frisbee"]))

    def test_predictions_carry_the_ground_truth_tubelets(self):
        prior = TripletPrior()
        prior.count(_doc())
        preds = frequency_predictions(_doc(), prior, top_k=1,
                                      max_per_video=100)
        p = [q for q in preds if q["triplet"][0] == "dog"][0]
        self.assertEqual(p["duration"], (0, 10))
        self.assertEqual(len(p["sub_traj"]), 10)
        self.assertEqual(p["sub_traj"][0], (0, 0, 9, 9))

    def test_the_cap_keeps_the_highest_scoring_predictions(self):
        prior = TripletPrior()
        prior.count(_doc())
        preds = frequency_predictions(_doc(), prior, top_k=5,
                                      max_per_video=2)
        self.assertEqual(len(preds), 2)
        self.assertGreaterEqual(preds[0]["score"], preds[1]["score"])

    def test_the_order_is_deterministic(self):
        prior = TripletPrior()
        prior.count(_doc())
        a = frequency_predictions(_doc(), prior, 5, 20)
        b = frequency_predictions(_doc(), prior, 5, 20)
        self.assertEqual([p["triplet"] for p in a],
                         [p["triplet"] for p in b])


class TestAssociation(unittest.TestCase):

    def test_adjacent_segments_of_one_triplet_become_one_instance(self):
        kept = [(0, "k", 0.8), (1, "k", 0.6), (2, "k", 0.4)]
        out = associate_segments(kept, n_frames=90, segment=30)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1:3], (0, 90))
        self.assertAlmostEqual(out[0][3], 0.6)

    def test_a_gap_breaks_the_association(self):
        kept = [(0, "k", 0.8), (2, "k", 0.4)]
        out = associate_segments(kept, n_frames=90, segment=30)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][1:3], (0, 30))
        self.assertEqual(out[1][1:3], (60, 90))

    def test_the_last_segment_stops_at_the_last_frame(self):
        kept = [(2, "k", 0.4)]
        out = associate_segments(kept, n_frames=70, segment=30)
        self.assertEqual(out[0][1:3], (60, 70))

    def test_different_triplets_never_merge(self):
        kept = [(0, "k", 0.8), (1, "j", 0.6)]
        out = associate_segments(kept, n_frames=60, segment=30)
        self.assertEqual(len(out), 2)


class TestDeadExports(unittest.TestCase):
    """Four exports under `eval/exports/` hold one all-zero code.

    A mAP computed on one of those describes a broken encoder, not a method,
    and it would look like an ordinary low score. The count has to be carried
    beside the number rather than checked once by hand.
    """

    def test_distinct_latents_counts_whole_codes_not_bits(self):
        z = np.array([[0, 0], [0, 0], [1, 0]], dtype=np.int8)
        self.assertEqual(distinct_latents(z), 2)

    def test_an_all_zero_export_holds_exactly_one_code(self):
        self.assertEqual(distinct_latents(np.zeros((40, 8), dtype=np.int8)), 1)

    def test_a_dead_export_is_dropped_rather_than_scored(self):
        from tools.planner.predicate_probe import relation_labels

        rng = np.random.RandomState(0)
        with tempfile.TemporaryDirectory() as tmp:
            clips = []
            for name in ("a", "b", "dead"):
                path = os.path.join(tmp, name + ".json")
                with open(path, "w") as f:
                    json.dump(_doc(name), f)
                z = (np.zeros((10, 8), dtype=np.int8) if name == "dead"
                     else rng.randint(0, 2, (10, 8)).astype(np.int8))
                clips.append((name, z, _doc(name),
                              relation_labels(path, num_objs=2)))
            _, diag = probe_predictions(clips, segment=5, top_k=2,
                                        max_per_video=10, test_frac=0.34)

        self.assertEqual(diag["n_dead_exports"], 1)
        self.assertEqual(diag["n_clips"], 2)
        self.assertGreater(diag["distinct_latents_min"], 1)


class TestRidgeAccumulator(unittest.TestCase):

    def test_it_matches_the_M1_probe_fitted_in_one_block(self):
        # M7's probe condition must score the SAME probe M1 reports, or the
        # two metrics are measuring different models and neither explains the
        # other. The accumulator exists only because stacking every clip's
        # rows needs about two gigabytes on VidOR.
        from tools.planner.predicate_probe import ridge_probe_multi

        rng = np.random.RandomState(0)
        x1, y1 = rng.rand(30, 5), rng.rand(30, 3)
        x2, y2 = rng.rand(20, 5), rng.rand(20, 3)
        xt = rng.rand(7, 5)

        acc = RidgeAccumulator(5, 3, alpha=1.0)
        acc.add(x1, y1)
        acc.add(x2, y2)
        got = acc.predict(acc.solve(), xt)

        want = ridge_probe_multi(np.vstack([x1, x2]), np.vstack([y1, y2]), xt)
        np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-8)


# ------------------------------------------------------------- the figures

class TestFigures(unittest.TestCase):

    def _parses(self, path):
        # A raw angle bracket inside text content is invalid XML, and this
        # project has shipped an unopenable figure that way three times.
        # `parse` is what catches it.
        xml.dom.minidom.parse(path)

    def test_a_result_figure_parses_as_xml(self):
        gt = {"a": relation_instances(_doc("a"))}
        r = evaluate(gt, {"a": perfect_predictions(gt["a"])})
        r["tag"] = "unit-test"
        r["dataset"] = "fixture"
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "map.svg")
            result_svg(r, p)
            self._parses(p)

    def test_a_comparison_figure_parses_as_xml(self):
        rows = [{"tag": "a-perfect", "mean_ap": 1.0, "dataset": "vidvrd",
                 "n_videos": 2},
                {"tag": "a-frequency", "mean_ap": 0.02, "dataset": "vidvrd",
                 "n_videos": 2}]
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m7_map.svg")
            comparison_svg(rows, p)
            self._parses(p)

    def test_a_label_holding_an_angle_bracket_is_escaped(self):
        rows = [{"tag": "a<b", "mean_ap": 0.5, "dataset": "x", "n_videos": 1}]
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m7_map.svg")
            comparison_svg(rows, p)
            xml.dom.minidom.parse(p)


# ---------------------------------------------------------------- the CLI

class TestCommandLine(unittest.TestCase):

    def test_a_perfect_run_end_to_end_writes_one_point_zero(self):
        from tools.planner.m7_map import main

        with tempfile.TemporaryDirectory() as tmp:
            ann = os.path.join(tmp, "ann")
            os.makedirs(ann)
            for name in ("a", "b"):
                with open(os.path.join(ann, name + ".json"), "w") as f:
                    json.dump(_doc(name), f)
            out = os.path.join(tmp, "out")
            code = main(["--annotations", os.path.join(ann, "*.json"),
                         "--predictor", "perfect",
                         "--tag", "fixture-perfect", "--out-dir", out])
            self.assertEqual(code, 0)
            with open(os.path.join(out, "fixture-perfect", "map.json")) as f:
                r = json.load(f)
            self.assertAlmostEqual(r["mean_ap"], 1.0)
            xml.dom.minidom.parse(
                os.path.join(out, "fixture-perfect", "map.svg"))


if __name__ == "__main__":
    unittest.main()
