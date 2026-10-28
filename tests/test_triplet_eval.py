#!/usr/bin/env python3
"""Tests for `triplet_eval.py`, the triplet mAP of the four rungs.

Every expected number comes from the published rule or from hand
arithmetic, never from a run of the code under test. The small scenarios
of the roadmap ("Έλεγχος με μικρά σενάρια") are the first class: a toy
clip in the VidVRD form, scored by the official evaluator itself. A test
that needs an evaluator clone skips when `data/deps/eval-src/` is absent.

    PYTHONPATH=. .venv-local/bin/python -m unittest \\
        tests.test_triplet_eval
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from metrics import triplet_eval as te    # noqa: E402

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


# ------------------------------------------------------------ the probe

class TestRelationCounts(unittest.TestCase):

    def test_counts_follow_bindings_and_bits(self):
        # U = 3, P = 2. Units 0 and 1 bind (0, 1) with predicate 0 on.
        # Unit 2 binds (1, 0) with predicate 1 on.
        latents = np.array([[1, 0, 1, 0, 0, 1]])
        bindings = np.array([[[0, 1], [0, 1], [1, 0]]])
        F = te.relation_counts(latents, bindings, P=2)
        self.assertEqual(F.shape, (1, 3, 3, 2))
        self.assertEqual(F[0, 0, 1].tolist(), [2.0, 0.0])
        self.assertEqual(F[0, 1, 0].tolist(), [0.0, 1.0])
        self.assertEqual(F.sum(), 3.0)


class TestProbeRows(unittest.TestCase):

    def setUp(self):
        latents, bindings, boxes = _rows()
        self.clip = {"frames": np.arange(10), "latents": latents,
                     "bindings": bindings, "gt_boxes": boxes}

    def test_one_row_for_each_frame_and_ordered_pair(self):
        keys, X, Y = te.probe_rows(_doc("a"), self.clip, 2, ["in", "on"])
        self.assertEqual(len(keys), 20)
        self.assertEqual(keys[0], (0, 0, 1))
        self.assertEqual(keys[1], (0, 1, 0))
        self.assertEqual(X.shape, (20, 2))
        self.assertEqual(Y.shape, (20, 2))

    def test_targets_mark_the_true_frames_of_the_pair(self):
        keys, X, Y = te.probe_rows(_doc("a"), self.clip, 2, ["in", "on"])
        on = [keys[i] for i in range(len(keys)) if Y[i, 1] == 1]
        self.assertEqual(on, [(t, 0, 1) for t in range(2, 6)])
        self.assertEqual(Y[:, 0].sum(), 0)

    def test_a_frame_where_one_track_has_no_box_has_no_row(self):
        doc = _doc("a")
        doc["trajectories"][7] = doc["trajectories"][7][:1]
        keys, _, _ = te.probe_rows(doc, self.clip, 2)
        self.assertEqual(len(keys), 18)
        self.assertNotIn(7, [k[0] for k in keys])


class TestReadouts(unittest.TestCase):

    def test_one_to_one_names_the_predicate_by_its_agreement(self):
        X = np.array([[1, 0], [1, 0], [1, 0], [0, 0]])
        Y = np.array([[0, 1], [0, 1], [1, 0], [0, 0]])
        names, agreement = te.one_to_one(X, Y)
        self.assertEqual(names.tolist(), [1, -1])
        self.assertAlmostEqual(agreement[0], 2.0 / 3.0)
        self.assertEqual(agreement[1], 0.0)

    def test_one_to_one_breaks_a_tie_by_vocabulary_order(self):
        names, _ = te.one_to_one(np.array([[1], [1]]),
                                 np.array([[0, 1], [1, 0]]))
        self.assertEqual(names.tolist(), [0])

    def test_one_to_one_scores_take_the_best_agreement(self):
        scores, thresholds = te.one_to_one_scores(
            np.array([[1, 1], [0, 1], [0, 0]]), np.array([1, 1]),
            np.array([0.4, 0.7]), 2)
        self.assertEqual(scores.tolist(), [[0, 0.7], [0, 0.7], [0, 0]])
        self.assertTrue(np.all(thresholds > 0))

    def test_the_ridge_threshold_matches_the_train_prevalence(self):
        X = np.array([[0.0], [0.0], [1.0], [1.0], [0.0], [1.0]])
        Y = np.array([[0.0], [0.0], [1.0], [1.0], [0.0], [0.0]])
        scores, thresholds = te.ridge_readout(X, Y, np.array([[1.0], [0.0]]))
        train = te.ridge_probe_multi(X, Y, X)
        self.assertAlmostEqual(thresholds[0], np.sort(train[:, 0])[-2], 8)
        self.assertGreaterEqual(scores[0, 0], thresholds[0])
        self.assertLess(scores[1, 0], thresholds[0])

    def test_ridge_scores_are_rounded_to_nine_decimals(self):
        X = np.random.RandomState(0).rand(50, 3)
        Y = (X[:, :1] > 0.5).astype(float)
        scores, thresholds = te.ridge_readout(X, Y, X[:5])
        self.assertTrue(np.array_equal(scores, np.round(scores, 9)))
        self.assertEqual(thresholds[0], np.round(thresholds[0], 9))

    def test_a_name_with_no_train_row_is_never_on(self):
        _, thresholds = te.ridge_readout(np.ones((3, 1)), np.zeros((3, 1)),
                                         np.ones((1, 1)))
        self.assertTrue(np.isinf(thresholds[0]))


class TestFramesToInstances(unittest.TestCase):

    def test_consecutive_frames_give_one_instance(self):
        keys = [(t, 0, 1) for t in range(10)]
        scores = np.zeros((10, 1))
        scores[2:6, 0] = [0.2, 0.4, 0.6, 0.8]
        out = te.frames_to_instances(_doc("a"), keys, scores, [0.1], ["on"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["triplet"], ("hand", "on", "cup"))
        self.assertEqual(out[0]["duration"], (2, 6))
        self.assertAlmostEqual(out[0]["score"], 0.5)
        self.assertEqual(out[0]["sub_traj"], [HAND] * 4)
        self.assertEqual(out[0]["obj_traj"], [CUP] * 4)

    def test_a_gap_gives_two_instances_high_score_first(self):
        keys = [(t, 0, 1) for t in range(10)]
        scores = np.zeros((10, 1))
        scores[[1, 2, 6], 0] = [0.3, 0.3, 0.9]
        out = te.frames_to_instances(_doc("a"), keys, scores, [0.1], ["on"])
        self.assertEqual([o["duration"] for o in out], [(6, 7), (1, 3)])

    def test_the_cap_keeps_the_best_instances(self):
        keys = [(t, 0, 1) for t in range(10)]
        scores = np.zeros((10, 1))
        scores[[0, 2, 4, 6], 0] = [0.1, 0.4, 0.3, 0.2]
        out = te.frames_to_instances(_doc("a"), keys, scores, [0.05], ["on"],
                                     max_per_video=2)
        self.assertEqual([o["duration"] for o in out], [(2, 3), (4, 5)])


class TestProbePredictions(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        spec = _toy_rung(tmp)
        self.split = te.read_split(spec["split"])
        self.export = te.read_export(spec["export"])
        self.docs = dict((c, _doc(c)) for c in self.export["clips"])

    def test_both_routes_recover_the_true_instance(self):
        for route in ("ridge", "onetoone"):
            pred, diag = te.probe_predictions(self.docs, self.export,
                                              self.split, ["on"], route)
            self.assertEqual(sorted(pred), ["s1", "v1"])
            self.assertEqual(len(pred["s1"]), 1, route)
            self.assertEqual(pred["s1"][0]["triplet"], ("hand", "on", "cup"))
            self.assertEqual(pred["s1"][0]["duration"], (2, 6))
            self.assertEqual(diag["train_rows"], 60)
            self.assertEqual(diag["train_positives"], {"on": 12})

    def test_the_one_to_one_map_is_in_the_diagnostics(self):
        _, diag = te.probe_predictions(self.docs, self.export, self.split,
                                       ["on"], "onetoone")
        self.assertEqual(diag["map"], {"0": "on", "1": None})
        self.assertEqual(diag["agreement"]["0"], 1.0)

    def test_an_unknown_route_is_refused(self):
        with self.assertRaises(ValueError):
            te.probe_predictions(self.docs, self.export, self.split, ["on"],
                                 "vlm")


# ------------------------------------------------------- the whole run

@NEED_BOTH
class TestRunDataset(unittest.TestCase):
    """The toy rung end to end, through both evaluator clones."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.spec = _toy_rung(cls.tmp)
        cls.out = os.path.join(cls.tmp, "out")
        cls.metrics = te.run_dataset(cls.spec, cls.out,
                                     te.pinned_evaluator(te.OFFICIAL),
                                     te.pinned_evaluator(te.CROSSCHECK))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def row(self, clipset, predictor):
        return te._row(self.metrics, clipset, predictor)

    def test_the_self_check_passes(self):
        for clipset in te.CLIPSETS:
            check = self.metrics["self_check"][clipset]
            self.assertTrue(check["pass"])
            self.assertEqual(check["perfect_uncut_mAP"], 1.0)
            self.assertEqual(check["perfect_cut_expected"], 1.0)
            self.assertEqual(check["clips_over_200"], 0)

    def test_perfect_and_both_probes_find_the_one_instance(self):
        for predictor in ("perfect", "ridge", "onetoone"):
            self.assertEqual(self.row("test", predictor)["official"]["mAP"],
                             1.0, predictor)

    def test_frequency_misses_on_time_and_ties_on_names(self):
        # Frequency covers frames [0, 10) and the truth [2, 6): vIoU 0.4.
        # Both orders of the pair get "on" with count 3. The tie goes to
        # ("cup", "on", "hand") by name, which is false, so P@1 = 0 and
        # P@5 = 1 / 2.
        got = self.row("test", "frequency")["official"]
        self.assertEqual(got["mAP"], 0.0)
        self.assertEqual(got["P@1"], 0.0)
        self.assertEqual(got["P@5"], 0.5)

    def test_the_cross_check_agrees_with_the_official_numbers(self):
        for predictor in te.PREDICTORS:
            row = self.row("test", predictor)
            self.assertEqual(row["crosscheck"], row["official"], predictor)

    def test_the_second_row_holds_test_and_validation_clips(self):
        self.assertEqual(self.row("test", "perfect")["n_clips"], 1)
        self.assertEqual(self.row("testval", "perfect")["n_clips"], 2)
        self.assertEqual(self.row("testval", "perfect")["clip_ap"],
                         {"s1": 1.0, "v1": 1.0})

    def test_the_prediction_files_have_the_official_form(self):
        for predictor in te.PREDICTORS:
            path = os.path.join(self.out,
                                "predictions_se_%s.json" % predictor)
            with open(path) as f:
                doc = json.load(f)
            self.assertEqual(doc["version"], "VERSION 1.0")
            self.assertEqual(sorted(doc["results"]), ["s1", "v1"])
            self.assertFalse(doc["external_data"]["used"])

    def test_the_metrics_file_records_the_run(self):
        with open(os.path.join(self.out, "metrics_se.json")) as f:
            m = json.load(f)
        self.assertEqual(m["clips"], {"train": 3, "val": 1, "test": 1})
        self.assertEqual(m["evaluator"]["commit"], te.OFFICIAL["commit"])
        self.assertEqual(m["crosscheck"]["commit"], te.CROSSCHECK["commit"])
        self.assertEqual(m["slot_boxes"],
                         {"checked": 100, "off_by_more_than_1px": 0})
        self.assertEqual(m["truth_dropped"], 0)
        self.assertEqual(m["vocabulary"], ["on"])
        self.assertIn(m["tqdm"], ("stub", "package"))
        self.assertEqual(len(m["rows"]), 8)

    def test_the_tables_hold_every_row(self):
        text = te.markdown_tables(self.out)
        self.assertIn("| SE | test | 1.0000 | 1.0000 | 1.0000 | 0 |", text)
        self.assertIn("| SE | test | ridge | 1.0000 |", text)
        self.assertIn("| SE | testval | frequency | +0.0000 |", text)
        self.assertIn("- SE, ridge: mAP 1.0000, above frequency 0.0000.",
                      text)


class TestRunRefusals(unittest.TestCase):

    def test_an_export_that_disagrees_with_the_split_stops_the_run(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        spec = _toy_rung(tmp)
        with open(spec["split"]) as f:
            doc = json.load(f)
        doc["video_ids"].append("t4")
        doc["split_info"]["clips"]["train"] = 4
        with open(spec["split"], "w") as f:
            json.dump(doc, f)
        with self.assertRaises(SystemExit):
            te.run_dataset(spec, os.path.join(tmp, "out"), (None, {}))

    def test_the_command_line_needs_a_dataset_to_score(self):
        with self.assertRaises(SystemExit):
            te.main(["--out", tempfile.gettempdir()])

    def test_tables_without_metrics_stop(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        with self.assertRaises(SystemExit):
            te.main(["--out", tmp, "--tables"])


# ------------------------------------------------------------- figures

def _have_matplotlib():
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


@NEED_BOTH
@unittest.skipUnless(_have_matplotlib(), "no matplotlib")
class TestFigures(unittest.TestCase):

    def test_the_figures_are_written_as_png(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        out = os.path.join(tmp, "out")
        spec = _toy_rung(tmp)
        te.run_dataset(spec, out, te.pinned_evaluator(te.OFFICIAL))
        paths = te.figures(out, os.path.join(tmp, "fig"))
        self.assertEqual([os.path.basename(p) for p in paths],
                         ["triplet-map-ladder.png", "triplet-clip-ap.png"])
        for path in paths:
            with open(path, "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
