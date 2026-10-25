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


# ------------------------------------------------- toy rung: data on disk

def _doc(video_id, n_frames=10, relations=((0, 1, "on", 2, 6),)):
    """A clip in the VidVRD form: a big hand (tid 0) and a small cup (tid 1).

    The hand has the larger area, so the slot map puts it in slot 0.
    """
    hand = dict(zip(("xmin", "ymin", "xmax", "ymax"), HAND))
    cup = dict(zip(("xmin", "ymin", "xmax", "ymax"), CUP))
    return {
        "video_id": video_id, "width": 640, "height": 360,
        "subject/objects": [{"tid": 0, "category": "hand"},
                            {"tid": 1, "category": "cup"}],
        "trajectories": [[{"tid": 0, "bbox": dict(hand)},
                          {"tid": 1, "bbox": dict(cup)}]
                         for _ in range(n_frames)],
        "relation_instances": [
            {"subject_tid": s, "object_tid": o, "predicate": p,
             "begin_fid": b, "end_fid": e}
            for s, o, p, b, e in relations],
    }


def _rows(n_frames=10, on_frames=range(2, 6)):
    """Export rows of one toy clip: U = 2 units, P = 2 predicates.

    Unit 0 binds slots (0, 1) and switches predicate 0 on at `on_frames`.
    Unit 1 binds slots (1, 0) and never switches a bit on.
    """
    latents = np.zeros((n_frames, 4), dtype=np.int8)
    for t in on_frames:
        latents[t, 0] = 1
    bindings = np.zeros((n_frames, 2, 2), dtype=np.int32)
    bindings[:, 0] = (0, 1)
    bindings[:, 1] = (1, 0)
    boxes = np.zeros((n_frames, 3, 4), dtype=np.float32)
    boxes[:, 0] = te.letterbox(HAND, 640, 360)
    boxes[:, 1] = te.letterbox(CUP, 640, 360)
    return latents, bindings, boxes


def _toy_rung(root, train=("t1", "t2", "t3"), val=("v1",), test=("s1",)):
    """Write a toy export, truth directory and split file. Return the spec."""
    truth = os.path.join(root, "truth")
    os.makedirs(truth)
    clips = list(train) + list(val) + list(test)
    ids, lat, bind, box = [], [], [], []
    for clip in clips:
        with open(os.path.join(truth, clip + ".json"), "w") as f:
            json.dump(_doc(clip), f)
        latents, bindings, boxes = _rows()
        ids += ["%s/%06d" % (clip, t) for t in range(len(latents))]
        lat.append(latents)
        bind.append(bindings)
        box.append(boxes)
    order = np.random.RandomState(0).permutation(len(ids))
    export = os.path.join(root, "toy_latents.npz")
    np.savez(export, latents=np.vstack(lat)[order],
             bindings=np.vstack(bind)[order], gt_boxes=np.vstack(box)[order],
             frame_ids=np.array(ids)[order], U=2, A=2, P=2, n_bits=4,
             model_name="toy")
    split = os.path.join(root, "loaded_videos.json")
    with open(split, "w") as f:
        json.dump({"video_ids": clips, "split_info": {
            "clips": {"train": len(train), "val": len(val),
                      "test": len(test)},
            "val_clips": list(val), "test_clips": list(test)}}, f)
    return {"name": "se", "export": export, "truth": truth, "split": split}


# ------------------------------------------- truth reader and predictors

class TestCopiedTruthAndPredictors(unittest.TestCase):

    def test_the_triplet_uses_category_names(self):
        insts = te.relation_instances(_doc("a"))
        self.assertEqual(insts[0]["triplet"], ("hand", "on", "cup"))
        self.assertEqual(insts[0]["duration"], (2, 6))
        self.assertEqual(len(insts[0]["sub_traj"]), 4)

    def test_an_instance_with_a_missing_box_drops(self):
        doc = _doc("a")
        doc["trajectories"][3] = doc["trajectories"][3][:1]
        self.assertEqual(te.relation_instances(doc), [])

    def test_perfect_returns_the_truth_with_score_one(self):
        insts = te.relation_instances(_doc("a"))
        pred = te.perfect_predictions(insts)
        self.assertEqual(pred[0]["score"], 1.0)
        self.assertEqual(pred[0]["triplet"], insts[0]["triplet"])

    def test_frequency_ranks_the_predicates_of_the_train_clips(self):
        prior = te.TripletPrior()
        prior.count(_doc("a", relations=((0, 1, "on", 0, 2),
                                         (0, 1, "on", 4, 6),
                                         (0, 1, "in", 0, 2))))
        pred = te.frequency_predictions(_doc("b"), prior, top_k=10)
        hand_first = [p for p in pred if p["triplet"][0] == "hand"]
        self.assertEqual([p["triplet"][1] for p in hand_first], ["on", "in"])
        self.assertEqual(hand_first[0]["score"], 2.0)
        self.assertEqual(hand_first[0]["duration"], (0, 10))


# ----------------------------------------------- split, export and slots

class TestSplitExportSlots(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.spec = _toy_rung(self.tmp)

    def test_the_train_clips_are_the_rest_of_video_ids(self):
        split = te.read_split(self.spec["split"])
        self.assertEqual(split, {"train": ["t1", "t2", "t3"],
                                 "val": ["v1"], "test": ["s1"]})

    def test_a_train_count_that_disagrees_stops_the_run(self):
        with open(self.spec["split"]) as f:
            doc = json.load(f)
        doc["split_info"]["clips"]["train"] = 2
        with open(self.spec["split"], "w") as f:
            json.dump(doc, f)
        with self.assertRaises(SystemExit):
            te.read_split(self.spec["split"])

    def test_the_export_is_cut_into_clips_in_frame_order(self):
        export = te.read_export(self.spec["export"])
        self.assertEqual(export["P"], 2)
        self.assertEqual(sorted(export["clips"]),
                         ["s1", "t1", "t2", "t3", "v1"])
        clip = export["clips"]["s1"]
        self.assertEqual(clip["frames"].tolist(), list(range(10)))
        self.assertEqual(clip["latents"][:, 0].tolist(),
                         [0, 0, 1, 1, 1, 1, 0, 0, 0, 0])

    def test_the_larger_track_takes_slot_zero(self):
        self.assertEqual(te.track_of_slot(_doc("a")), {0: 0, 1: 1})

    def test_letterbox_of_a_full_640_by_360_frame(self):
        # s = 0.46875 and the bars are 15.625 high: hand arithmetic.
        self.assertEqual(te.letterbox((0, 0, 639, 359), 640, 360),
                         (0, 16, 299, 184))

    def test_the_slot_check_finds_a_moved_box(self):
        clip = te.read_export(self.spec["export"])["clips"]["s1"]
        self.assertEqual(te.slot_box_mismatches(_doc("s1"), clip), (20, 0))
        clip["gt_boxes"][4, 1] += 5
        self.assertEqual(te.slot_box_mismatches(_doc("s1"), clip), (20, 1))
