"""A reconstruction loss means nothing without the loss of a trivial model.

The author judges the rung runs by eye from `training_curve.png`. A val BCE of
0.12 says nothing alone, so each run writes `baseline.json`: the BCE and MSE a
model would score if it always predicted the mean training state. A curve
under that line learned more than the average frame. Added 2026-10-03, plan
`notes/plans/2026-10-03-train-the-rungs.md` step 1.0b.

The model's `BCE` metric is Keras `binary_crossentropy` over the flattened
state, clipped to [1e-7, 1 - 1e-7], averaged over features and then over
samples (`latplan/util/distances.py`). The baseline is scored the same way.
"""

import importlib.util
import math
import os
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))

_spec = importlib.util.spec_from_file_location(
    "_baseline", os.path.join(ROOT, "latplan", "util", "baseline.py"))
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


class TestMeanBaseline(unittest.TestCase):

    def test_a_half_mean_against_zeros_and_ones_scores_log_two_and_a_quarter(self):
        train = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        val = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        got = baseline.mean_baseline(train, val)
        self.assertAlmostEqual(got["val_BCE"], math.log(2), places=5)
        self.assertAlmostEqual(got["val_MSE"], 0.25, places=6)

    def test_states_with_object_slots_are_flattened_before_scoring(self):
        train = np.full((4, 3, 2), 0.5, dtype=np.float32)
        val = np.zeros((2, 3, 2), dtype=np.float32)
        got = baseline.mean_baseline(train, val)
        self.assertAlmostEqual(got["val_BCE"], math.log(2), places=5)
        self.assertAlmostEqual(got["val_MSE"], 0.25, places=6)

    def test_a_perfect_mean_is_clipped_rather_than_infinite(self):
        train = np.ones((3, 2), dtype=np.float32)
        val = np.zeros((1, 2), dtype=np.float32)
        got = baseline.mean_baseline(train, val)
        self.assertTrue(math.isfinite(got["val_BCE"]))
        self.assertAlmostEqual(got["val_BCE"], -math.log(1e-7), places=2)

    def test_the_training_split_is_scored_too(self):
        train = np.array([[0.0], [1.0]], dtype=np.float32)
        got = baseline.mean_baseline(train, train)
        self.assertAlmostEqual(got["train_BCE"], math.log(2), places=5)
        self.assertIn("rule", got)


if __name__ == "__main__":
    unittest.main()
