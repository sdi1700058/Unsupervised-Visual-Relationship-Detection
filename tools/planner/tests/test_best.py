"""Keep the weights of the best val epoch, and score held-out states.

Found 2026-10-06: every run saved its weights at the last epoch only, while
its val BCE was lowest near epoch 1,000 of 5,000 and rose after, so the best
weights could not be tested. `latplan/util/best.py` and the scores in
`latplan/util/baseline.py` import nothing from `latplan`, so they load here
without TensorFlow.
"""

import importlib.util
import math
import os
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


best = _load("_best", "latplan/util/best.py")
baseline = _load("_baseline_b", "latplan/util/baseline.py")


class BestKeeper(unittest.TestCase):

    def test_the_lowest_value_keeps_its_weights_and_its_epoch(self):
        keeper = best.BestKeeper()
        for epoch, value in enumerate([0.9, 0.5, 0.7, 0.4, 0.6]):
            keeper.update(epoch, value, lambda e=epoch: [np.full(2, e)])
        self.assertEqual(keeper.epoch, 3)
        self.assertAlmostEqual(keeper.value, 0.4)
        np.testing.assert_array_equal(keeper.weights[0], [3, 3])

    def test_the_kept_weights_are_a_copy(self):
        keeper = best.BestKeeper()
        w = [np.zeros(2)]
        keeper.update(0, 0.5, lambda: w)
        w[0][:] = 7
        np.testing.assert_array_equal(keeper.weights[0], [0, 0])

    def test_nothing_kept_before_the_first_update(self):
        keeper = best.BestKeeper()
        self.assertIsNone(keeper.epoch)
        self.assertIsNone(keeper.weights)

    def test_a_value_that_is_not_a_number_is_skipped(self):
        keeper = best.BestKeeper()
        keeper.update(0, float("nan"), lambda: [np.zeros(1)])
        keeper.update(1, 0.3, lambda: [np.ones(1)])
        self.assertEqual(keeper.epoch, 1)


class FromEpoch(unittest.TestCase):
    """The val BCE is high and unstable until the pre-encoder fade-in ends,
    and spikes at its end, so the keeper counts only from then on. Found
    2026-10-06 by an adversary on the runs of 5,000 epochs."""

    def test_an_epoch_before_the_start_is_not_kept(self):
        keeper = best.BestKeeper(start=3)
        for epoch, value in enumerate([0.1, 0.2, 0.9, 0.5, 0.6]):
            keeper.update(epoch, value, lambda e=epoch: [np.full(1, e)])
        self.assertEqual(keeper.epoch, 3)


class SplitByCounts(unittest.TestCase):

    def test_flat_weights_go_back_to_each_network_in_order(self):
        flat = [np.zeros(1), np.ones(1), np.full(1, 2.0)]
        parts = best.split_by_counts(flat, [2, 1])
        self.assertEqual([len(p) for p in parts], [2, 1])
        np.testing.assert_array_equal(parts[1][0], [2.0])

    def test_counts_that_do_not_add_up_are_refused(self):
        with self.assertRaises(ValueError):
            best.split_by_counts([np.zeros(1)], [2])


class ReconstructionScores(unittest.TestCase):

    def test_object_states_of_three_axes_are_flattened(self):
        x = np.zeros((2, 3, 4))
        y = np.full((2, 3, 4), 0.5)
        got = baseline.reconstruction_scores(x, y)
        self.assertAlmostEqual(got["BCE"], math.log(2), places=6)
        self.assertAlmostEqual(got["MSE"], 0.25)


class ReconstructionScoresTwoAxes(unittest.TestCase):

    def test_a_perfect_copy_of_binary_states_scores_near_zero(self):
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        got = baseline.reconstruction_scores(x, x.copy())
        self.assertLess(got["BCE"], 1e-6)
        self.assertEqual(got["MSE"], 0.0)

    def test_a_half_prediction_scores_log_two(self):
        x = np.array([[0.0, 1.0], [1.0, 0.0]])
        got = baseline.reconstruction_scores(x, np.full_like(x, 0.5))
        self.assertAlmostEqual(got["BCE"], math.log(2), places=6)
        self.assertAlmostEqual(got["MSE"], 0.25)


if __name__ == "__main__":
    unittest.main()
