#!/usr/bin/env python3
"""Tests for `triplet_eval.py`, the triplet mAP of the four rungs.

Every expected number comes from the published rule or from hand
arithmetic, never from a run of the code under test. The small scenarios
of the roadmap ("Έλεγχος με μικρά σενάρια") are the first class: a toy
clip in the VidVRD form, scored by the official evaluator itself. A test
that needs an evaluator clone skips when `data/deps/eval-src/` is absent.

    PYTHONPATH=. .venv-local/bin/python -m unittest \\
        tools.planner.tests.test_triplet_eval
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from tools.planner import triplet_eval as te    # noqa: E402

HAVE_OFFICIAL = os.path.isfile(os.path.join(
    te.EVAL_SRC, te.OFFICIAL["dir"], "evaluation",
    te.OFFICIAL["module"] + ".py"))
HAVE_CROSSCHECK = os.path.isfile(os.path.join(
    te.EVAL_SRC, te.CROSSCHECK["dir"], "evaluation",
    te.CROSSCHECK["module"] + ".py"))
NEED_OFFICIAL = unittest.skipUnless(HAVE_OFFICIAL, "no official clone")
NEED_BOTH = unittest.skipUnless(HAVE_OFFICIAL and HAVE_CROSSCHECK,
                                "no evaluator clones")

HAND = (0, 0, 19, 19)
CUP = (30, 30, 39, 39)


def _inst(triplet, duration, sub=HAND, obj=CUP, score=None):
    """One relation instance with constant boxes over its duration."""
    n = duration[1] - duration[0]
    out = {"triplet": tuple(triplet), "duration": tuple(duration),
           "sub_traj": [tuple(sub)] * n, "obj_traj": [tuple(obj)] * n}
    if score is not None:
        out["score"] = score
    return out


# ------------------------------------------------------- small scenarios

@NEED_OFFICIAL
class TestSmallScenarios(unittest.TestCase):
    """The official evaluator on toy clips whose answer is known."""

    @classmethod
    def setUpClass(cls):
        cls.evaluate = staticmethod(te.load_evaluator(te.OFFICIAL))

    def official(self, gt, pred, top_returns=200):
        return te.score(self.evaluate, gt, pred, top_returns=top_returns)

    def test_perfect_gives_one(self):
        gt = {"c": [_inst(("hand", "on", "cup"), (2, 6))]}
        pred = {"c": [_inst(("hand", "on", "cup"), (2, 6), score=1.0)]}
        for cut in (200, None):
            got = self.official(gt, pred, cut)
            self.assertEqual(got, {"mAP": 1.0, "R@50": 1.0, "R@100": 1.0,
                                   "P@1": 1.0, "P@5": 1.0, "P@10": 1.0})

    def test_no_prediction_gives_zero(self):
        gt = {"c": [_inst(("hand", "on", "cup"), (2, 6))]}
        got = self.official(gt, {"c": []})
        self.assertEqual(set(got.values()), {0.0})

    def test_a_wrong_predicate_gives_zero(self):
        gt = {"c": [_inst(("hand", "on", "cup"), (2, 6))]}
        pred = {"c": [_inst(("hand", "in", "cup"), (2, 6), score=1.0)]}
        self.assertEqual(self.official(gt, pred)["mAP"], 0.0)

    def test_an_unnamed_model_predicate_gives_zero(self):
        # FOSAE calls its predicates pred_<p> (fol.py:100). Without a name
        # map no predicted triplet equals a true one.
        gt = {"c": [_inst(("hand", "on", "cup"), (2, 6))]}
        pred = {"c": [_inst(("hand", "pred_0", "cup"), (2, 6), score=1.0)]}
        got = self.official(gt, pred)
        self.assertEqual(got["mAP"], 0.0)
        self.assertEqual(got["P@1"], 0.0)

    def test_a_wrong_category_gives_zero(self):
        gt = {"c": [_inst(("hand", "on", "cup"), (2, 6))]}
        pred = {"c": [_inst(("cup", "on", "cup"), (2, 6), score=1.0)]}
        self.assertEqual(self.official(gt, pred)["mAP"], 0.0)

    def test_a_two_frame_prediction_of_a_one_frame_truth_still_hits(self):
        # Constant boxes: vIoU = 1 / 2 = 0.5, and the rule is ov >= 0.5.
        gt = {"c": [_inst(("hand", "on", "cup"), (3, 4))]}
        pred = {"c": [_inst(("hand", "on", "cup"), (3, 5), score=1.0)]}
        self.assertEqual(self.official(gt, pred)["mAP"], 1.0)

    def test_a_three_frame_prediction_of_a_one_frame_truth_misses(self):
        # vIoU = 1 / 3, below 0.5. Tagging ignores time, so P@1 stays 1.
        gt = {"c": [_inst(("hand", "on", "cup"), (3, 4))]}
        pred = {"c": [_inst(("hand", "on", "cup"), (3, 6), score=1.0)]}
        got = self.official(gt, pred)
        self.assertEqual(got["mAP"], 0.0)
        self.assertEqual(got["P@1"], 1.0)

    def test_a_prediction_one_frame_late_misses(self):
        gt = {"c": [_inst(("hand", "on", "cup"), (3, 4))]}
        pred = {"c": [_inst(("hand", "on", "cup"), (4, 5), score=1.0)]}
        self.assertEqual(self.official(gt, pred)["mAP"], 0.0)

    def test_the_cut_of_200_caps_a_clip_of_300_at_two_thirds(self):
        # AP = 200 / 300 under the cut (report B.4, self-check).
        gt = {"c": [_inst(("hand", "p%03d" % i, "cup"), (0, 1))
                    for i in range(300)]}
        pred = {"c": [dict(g, score=1.0) for g in gt["c"]]}
        self.assertEqual(self.official(gt, pred, 200)["mAP"], 0.6667)
        self.assertEqual(self.official(gt, pred, None)["mAP"], 1.0)

    def test_a_clip_with_no_truth_is_skipped(self):
        gt = {"a": [_inst(("hand", "on", "cup"), (2, 6))], "b": []}
        pred = {"a": [_inst(("hand", "on", "cup"), (2, 6), score=1.0)],
                "b": [_inst(("hand", "in", "cup"), (2, 6), score=1.0)]}
        self.assertEqual(self.official(gt, pred)["mAP"], 1.0)

    def test_the_mean_runs_over_clips(self):
        gt = {"a": [_inst(("hand", "on", "cup"), (2, 6))],
              "b": [_inst(("hand", "on", "cup"), (2, 6))]}
        pred = {"a": [_inst(("hand", "on", "cup"), (2, 6), score=1.0)],
                "b": []}
        self.assertEqual(self.official(gt, pred)["mAP"], 0.5)

    def test_clip_aps_gives_the_ap_of_each_clip(self):
        gt = {"a": [_inst(("hand", "on", "cup"), (2, 6))],
              "b": [_inst(("hand", "on", "cup"), (2, 6))], "c": []}
        pred = {"a": [_inst(("hand", "on", "cup"), (2, 6), score=1.0)]}
        got = te.clip_aps(self.evaluate, gt, pred, top_returns=200)
        self.assertEqual(got, {"a": 1.0, "b": 0.0})


# ----------------------------------------------------- loading the clones

class TestEvaluatorLoading(unittest.TestCase):

    def test_the_stub_makes_tqdm_importable(self):
        te.install_tqdm_stub()
        import tqdm
        self.assertEqual(list(tqdm.tqdm([1, 2])), [1, 2])

    @NEED_BOTH
    def test_both_clones_load_side_by_side(self):
        official = te.load_evaluator(te.OFFICIAL)
        cross = te.load_evaluator(te.CROSSCHECK)
        self.assertIsNot(official, cross)
        self.assertIn("top_returns", official.__code__.co_varnames)
        self.assertNotIn("top_returns", cross.__code__.co_varnames)

    @NEED_BOTH
    def test_the_cross_check_agrees_on_a_perfect_toy(self):
        gt = {"c": [_inst(("hand", "on", "cup"), (2, 6))]}
        pred = {"c": [_inst(("hand", "on", "cup"), (2, 6), score=1.0)]}
        official = te.score(te.load_evaluator(te.OFFICIAL), gt, pred)
        cross = te.score(te.load_evaluator(te.CROSSCHECK), gt, pred)
        self.assertEqual(official, cross)

    @NEED_OFFICIAL
    def test_the_official_clone_sits_at_its_pinned_commit(self):
        evaluate, record = te.pinned_evaluator(te.OFFICIAL)
        self.assertEqual(record["commit"], te.OFFICIAL["commit"])
        self.assertTrue(record["clean"])

    @NEED_OFFICIAL
    def test_a_clone_at_another_commit_is_refused(self):
        spec = dict(te.OFFICIAL, commit="0" * 40)
        with self.assertRaises(SystemExit):
            te.pinned_evaluator(spec)

    def test_a_missing_clone_is_refused(self):
        spec = dict(te.OFFICIAL, dir="no_such_clone")
        with self.assertRaises(SystemExit):
            te.load_evaluator(spec)
