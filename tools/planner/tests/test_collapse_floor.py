"""The analytic floor of the reconstruction loss, and what sits on it.

Sixteen runs of the overnight sweep landed between 0.488 and 0.597 whatever
`ZEROSUPPRESS`, `MAX_TEMPERATURE` or `U/A/P` were set to. That is not a
plateau. `val_loss = 0.5245` is `H(0.2182)`, the entropy of the mean of the
data, which is the loss of a decoder that has learned to emit one constant.

The loss is BCE averaged over the whole flattened feature vector
(`latplan/util/distances.py`), so its floor is the density of the data and does
not depend on `U`, `A` or `P` at all. That is exactly why every architecture
converged to the same place.

This module computes that floor from a baked dataset without going near
TensorFlow, so a run can be checked before any GPU time is spent on it.
Measured on CPU over 12 VidVRD videos at 30fps: `mo=5, patch=32` gives a
scalar-mean floor of 0.5235, against `H(0.2182) = 0.5246`.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    os.pardir)))

from tools.planner import collapse_floor as cf


class TestEntropy(unittest.TestCase):
    """In nats, because Keras BCE uses the natural log."""

    def test_the_measured_case(self):
        """The number this whole module exists to explain."""
        self.assertAlmostEqual(cf.entropy(0.2182), 0.5246, places=4)

    def test_a_fair_coin_is_the_maximum(self):
        self.assertAlmostEqual(cf.entropy(0.5), math.log(2), places=10)
        for p in (0.1, 0.3, 0.7, 0.9):
            self.assertLess(cf.entropy(p), cf.entropy(0.5))

    def test_a_certain_bit_costs_nothing(self):
        self.assertEqual(cf.entropy(0.0), 0.0)
        self.assertEqual(cf.entropy(1.0), 0.0)

    def test_it_is_symmetric(self):
        self.assertAlmostEqual(cf.entropy(0.2182), cf.entropy(1 - 0.2182),
                               places=12)

    def test_it_is_nats_and_not_bits(self):
        """A one-line slip that would move every verdict."""
        self.assertNotAlmostEqual(cf.entropy(0.5), 1.0, places=6)


class TestFeatureMean(unittest.TestCase):
    """`p` over the concatenated feature vector, padding included.

    The layout is `[patch**2 * 3 | x1(60) | y1(40) | x2(60) | y2(40)]`. A real
    object contributes exactly four ones to the 200-dim box block; a padded
    slot contributes an all-zero patch and an all-zero box.
    """

    def _clip(self, n_frames, n_objs, patch, fill, real):
        images = np.zeros((n_frames, n_objs, patch, patch, 3), dtype=np.float32)
        boxes = np.zeros((n_frames, n_objs, 4), dtype=np.float32)
        images[:, :real] = fill
        boxes[:, :real] = np.array([1.0, 2.0, 3.0, 4.0])
        return images, boxes

    def test_an_all_zero_clip_has_mean_zero(self):
        images, boxes = self._clip(2, 3, 4, 0.0, 0)
        self.assertEqual(cf.feature_mean(images, boxes), 0.0)

    def test_a_real_object_contributes_four_box_bits(self):
        """One slot, no patch signal: 4 ones out of patch_dim + 200."""
        images, boxes = self._clip(1, 1, 4, 0.0, 1)
        patch_dim = 4 * 4 * 3
        self.assertAlmostEqual(cf.feature_mean(images, boxes),
                               4.0 / (patch_dim + 200), places=10)

    def test_padding_dilutes_the_mean(self):
        """The reason more object slots means a lower floor."""
        tight = cf.feature_mean(*self._clip(1, 1, 4, 0.5, 1))
        padded = cf.feature_mean(*self._clip(1, 5, 4, 0.5, 1))
        self.assertLess(padded, tight)

    def test_the_patch_dominates_at_a_large_patch(self):
        """Patch 32 gives the box block 6.1 percent of the vector."""
        images, boxes = self._clip(1, 1, 32, 1.0, 1)
        patch_dim = 32 * 32 * 3
        self.assertAlmostEqual(cf.feature_mean(images, boxes),
                               (patch_dim + 4.0) / (patch_dim + 200),
                               places=10)

    def test_the_box_block_dominates_at_a_small_patch(self):
        share = cf.box_share(8)
        self.assertAlmostEqual(share, 200.0 / (8 * 8 * 3 + 200), places=6)
        self.assertGreater(share, 0.5)
        self.assertLess(cf.box_share(32), 0.07)

    def test_the_share_falls_as_the_patch_grows(self):
        """Slice G and slice H measured 51.0 and 6.1 percent independently."""
        self.assertAlmostEqual(cf.box_share(8) * 100, 51.02, places=1)
        self.assertAlmostEqual(cf.box_share(16) * 100, 20.66, places=1)
        self.assertAlmostEqual(cf.box_share(32) * 100, 6.11, places=1)

    def test_a_wrong_shape_raises_rather_than_guessing(self):
        self.assertRaises(ValueError, cf.feature_mean,
                          np.zeros((2, 3)), np.zeros((2, 3, 4)))


class TestVerdict(unittest.TestCase):
    """Whether a run is sitting on the floor."""

    def test_a_run_at_the_floor_is_named_dead(self):
        v = cf.verdict(0.5245, 0.5246)
        self.assertEqual(v["state"], "at_the_floor")
        self.assertIn("constant", v["reading"].lower())

    def test_a_run_well_below_the_floor_has_learned_something(self):
        self.assertEqual(cf.verdict(0.11, 0.5246)["state"], "below_the_floor")

    def test_a_run_above_the_floor_is_worse_than_a_constant(self):
        """Worse than emitting the mean, which is its own kind of broken."""
        self.assertEqual(cf.verdict(0.9, 0.5246)["state"], "above_the_floor")

    def test_the_tolerance_is_reported_not_hidden(self):
        self.assertIn("tolerance", cf.verdict(0.5245, 0.5246))

    def test_a_missing_loss_gives_no_verdict_rather_than_a_guess(self):
        self.assertIsNone(cf.verdict(None, 0.5246)["state"])

    def test_the_gap_is_signed_so_the_direction_is_readable(self):
        self.assertLess(cf.verdict(0.11, 0.5246)["gap"], 0)
        self.assertGreater(cf.verdict(0.9, 0.5246)["gap"], 0)


class TestReadingAHistory(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def _history(self, rows):
        path = os.path.join(self.dir, "training_history.csv")
        with open(path, "w") as handle:
            handle.write("epoch,BCE,val_BCE\n")
            for i, (a, b) in enumerate(rows):
                handle.write("%d,%s,%s\n" % (i, a, b))
        return path

    def test_it_takes_the_minimum_validation_loss(self):
        p = self._history([(0.9, 0.8), (0.7, 0.53), (0.6, 0.61)])
        self.assertAlmostEqual(cf.val_bce_min(p), 0.53, places=6)

    def test_a_missing_history_gives_none_rather_than_raising(self):
        self.assertIsNone(cf.val_bce_min(os.path.join(self.dir, "no.csv")))

    def test_a_history_with_no_rows_gives_none(self):
        self.assertIsNone(cf.val_bce_min(self._history([])))

    def test_an_unparsable_value_is_skipped_not_counted_as_zero(self):
        """A zero would read as the best run ever recorded."""
        p = self._history([(0.9, "nan"), (0.7, 0.53)])
        self.assertAlmostEqual(cf.val_bce_min(p), 0.53, places=6)


if __name__ == "__main__":
    unittest.main()
