#!/usr/bin/env python3
"""Tests for M1, the predicate probe.

Pure functions only: label extraction, the temporal split, average precision
and the probes themselves on synthetic data where the right answer is known by
construction. Nothing here needs a model, a dataset or a network.

    python3 -m unittest tools/planner/tests/test_predicate_probe.py
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _ann(tmp):
    """A two-object clip with two relations over known frame ranges."""
    doc = {
        "video_id": "TEST_0001",
        "width": 640, "height": 360, "fps": 30, "frame_count": 10,
        "subject/objects": [{"tid": 0, "category": "dog"},
                            {"tid": 1, "category": "frisbee"}],
        "relation_instances": [
            {"subject_tid": 0, "object_tid": 1, "predicate": "chase",
             "begin_fid": 0, "end_fid": 5},
            {"subject_tid": 0, "object_tid": 1, "predicate": "larger_than",
             "begin_fid": 0, "end_fid": 10},
            {"subject_tid": 1, "object_tid": 0, "predicate": "away",
             "begin_fid": 5, "end_fid": 10},
        ],
        "trajectories": [
            [{"tid": 0, "bbox": {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}},
             {"tid": 1, "bbox": {"xmin": 20, "ymin": 20, "xmax": 25, "ymax": 25}}]
            for _ in range(10)
        ],
    }
    p = Path(tmp) / "TEST_0001.json"
    p.write_text(json.dumps(doc))
    return str(p)


class TestLabels(unittest.TestCase):

    def test_relation_spans_become_per_frame_labels(self):
        from tools.planner.predicate_probe import relation_labels

        with tempfile.TemporaryDirectory() as tmp:
            lab = relation_labels(_ann(tmp), num_objs=2)

        # `end_fid` is exclusive in VidVRD, so "chase" covers frames 0..4.
        self.assertIn("chase", lab.predicates)
        self.assertEqual(lab.Y.shape[0], lab.pairs.shape[0])
        self.assertEqual(lab.Y.shape[1], len(lab.predicates))

        ci = lab.predicates.index("chase")
        rows_01 = [k for k in range(len(lab.frames))
                   if tuple(lab.pairs[k]) == (0, 1)]
        on = [lab.frames[k] for k in rows_01 if lab.Y[k, ci] > 0]
        self.assertEqual(on, [0, 1, 2, 3, 4])

    def test_direction_matters(self):
        """(0,1) and (1,0) are different rows; a probe must not conflate them."""
        from tools.planner.predicate_probe import relation_labels

        with tempfile.TemporaryDirectory() as tmp:
            lab = relation_labels(_ann(tmp), num_objs=2)

        ai = lab.predicates.index("away")
        for k in range(len(lab.frames)):
            if lab.Y[k, ai] > 0:
                self.assertEqual(tuple(lab.pairs[k]), (1, 0))

    def test_a_frame_carries_several_predicates_at_once(self):
        from tools.planner.predicate_probe import relation_labels

        with tempfile.TemporaryDirectory() as tmp:
            lab = relation_labels(_ann(tmp), num_objs=2)

        rows = [k for k in range(len(lab.frames))
                if lab.frames[k] == 2 and tuple(lab.pairs[k]) == (0, 1)]
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(lab.Y[rows[0]].sum()), 2)   # chase + larger_than


class TestSplit(unittest.TestCase):

    def test_split_is_temporal_not_random(self):
        """Adjacent frames are near-duplicates, so a random split leaks."""
        from tools.planner.predicate_probe import temporal_split

        frames = np.repeat(np.arange(100), 2)
        tr, te = temporal_split(frames, test_frac=0.3)
        self.assertTrue(frames[tr].max() < frames[te].min())

    def test_no_frame_appears_on_both_sides(self):
        from tools.planner.predicate_probe import temporal_split

        frames = np.repeat(np.arange(50), 3)
        tr, te = temporal_split(frames, test_frac=0.4)
        self.assertEqual(set(frames[tr]) & set(frames[te]), set())


class TestAveragePrecision(unittest.TestCase):

    def test_perfect_ranking_scores_one(self):
        from tools.planner.predicate_probe import average_precision

        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(average_precision(y, s), 1.0)

    def test_all_one_class_is_undefined(self):
        from tools.planner.predicate_probe import average_precision

        self.assertIsNone(average_precision(np.zeros(5), np.arange(5)))
        self.assertIsNone(average_precision(np.ones(5), np.arange(5)))

    def test_reversed_ranking_scores_near_the_floor(self):
        from tools.planner.predicate_probe import average_precision

        y = np.array([1, 1, 0, 0, 0, 0])
        good = average_precision(y, np.array([1.0, .9, .1, .1, .1, .1]))
        bad = average_precision(y, np.array([.1, .1, 1.0, .9, .8, .7]))
        self.assertGreater(good, bad)


class TestProbes(unittest.TestCase):

    def _separable(self, n=200, bits=16, seed=0):
        """Latents where bit 0 IS the label, plus noise bits."""
        rng = np.random.RandomState(seed)
        X = rng.randint(0, 2, size=(n, bits)).astype(np.float64)
        y = X[:, 0].copy()
        return X, y

    def test_ridge_probe_recovers_a_linearly_encoded_predicate(self):
        from tools.planner.predicate_probe import ridge_probe, average_precision

        X, y = self._separable()
        s = ridge_probe(X[:150], y[:150], X[150:])
        self.assertGreater(average_precision(y[150:], s), 0.95)

    def test_ridge_probe_fails_on_a_label_the_latent_does_not_carry(self):
        from tools.planner.predicate_probe import ridge_probe, average_precision

        X, _ = self._separable()
        rng = np.random.RandomState(7)
        y = rng.randint(0, 2, size=len(X))       # independent of X
        s = ridge_probe(X[:150], y[:150], X[150:])
        ap = average_precision(y[150:], s)
        self.assertLess(ap, 0.75)

    def test_knn_probe_finds_a_nonlinearly_encoded_predicate(self):
        """XOR of two bits: present in the code, not linearly decodable.

        This is the whole reason both probes exist. A low ridge score with a
        high kNN score means the relation IS there but is not *expressed*,
        which is a different and weaker claim than "the model learned it".
        """
        from tools.planner.predicate_probe import (ridge_probe, knn_probe,
                                                   average_precision)

        rng = np.random.RandomState(3)
        X = rng.randint(0, 2, size=(400, 8)).astype(np.float64)
        y = (X[:, 0].astype(int) ^ X[:, 1].astype(int))

        lin = average_precision(y[300:], ridge_probe(X[:300], y[:300], X[300:]))
        nn = average_precision(y[300:], knn_probe(X[:300], y[:300], X[300:], k=1))
        self.assertLess(lin, 0.75)
        self.assertGreater(nn, 0.9)


class TestShuffleControl(unittest.TestCase):

    def test_shuffled_latents_destroy_a_real_signal(self):
        """The control that makes the headline number mean anything.

        A probe can score well by learning the label's base rate and which
        object pair it is looking at, with no help from the latent at all.
        Shuffling the latents across frames keeps both of those and removes
        only the representation, so the gap between the two is the part the
        representation is responsible for.
        """
        from tools.planner.predicate_probe import (ridge_probe,
                                                   average_precision)

        rng = np.random.RandomState(0)
        X = rng.randint(0, 2, size=(300, 12)).astype(np.float64)
        y = X[:, 0].copy()

        real = average_precision(y[200:], ridge_probe(X[:200], y[:200], X[200:]))

        perm = rng.permutation(len(X))
        Xs = X[perm]
        shuffled = average_precision(
            y[200:], ridge_probe(Xs[:200], y[:200], Xs[200:]))

        self.assertGreater(real, 0.95)
        self.assertLess(shuffled, 0.8)
        self.assertGreater(real - shuffled, 0.2)


if __name__ == "__main__":
    unittest.main()
