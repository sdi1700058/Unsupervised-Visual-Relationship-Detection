#!/usr/bin/env python3
"""Tests for the temporal-fidelity screen (SPEC V35).

The screen asks whether latent path length tracks real time. It exists because
`latent_geometry` does not: geometry scores distance ORDERING among observed
frames, planning needs PATH LENGTH between them, and on H14 only the second
predicted planner error.

    python3 -m unittest tools/planner/tests/test_temporal_fidelity.py
"""

import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _one_bit_per_frame(n=40, bits=64):
    """A code where consecutive frames differ by exactly one new bit.

    Temporal distance is preserved perfectly: frames k apart are k observed
    steps apart. This is the oracle's behaviour.
    """
    z = np.zeros((n, bits), dtype=np.int8)
    for i in range(1, n):
        z[i] = z[i - 1]
        z[i, i % bits] = 1
    return z


def _compressed(n=40, bits=64, every=3):
    """A code that only changes every `every` frames.

    Frames k apart are about k/every steps apart, so the code compresses time
    and a planner will arrive before the window is filled.
    """
    z = np.zeros((n, bits), dtype=np.int8)
    for i in range(1, n):
        z[i] = z[i - 1]
        if i % every == 0:
            z[i, (i // every) % bits] = 1
    return z


class TestTemporalFidelity(unittest.TestCase):

    def test_a_code_that_preserves_time_scores_one(self):
        from tools.planner.latent_geometry import temporal_fidelity

        r = temporal_fidelity(_one_bit_per_frame(), ks=(2, 4, 7))
        self.assertAlmostEqual(r["steps_per_frame"], 1.0, places=1)
        self.assertEqual(r["per_k"][7], 7.0)

    def test_a_compressed_code_scores_below_one(self):
        from tools.planner.latent_geometry import temporal_fidelity

        r = temporal_fidelity(_compressed(every=3), ks=(2, 4, 7))
        self.assertLess(r["steps_per_frame"], 0.8)

    def test_compression_is_ordered_by_severity(self):
        from tools.planner.latent_geometry import temporal_fidelity

        mild = temporal_fidelity(_compressed(every=2), ks=(4, 7))
        harsh = temporal_fidelity(_compressed(every=5), ks=(4, 7))
        self.assertGreater(mild["steps_per_frame"], harsh["steps_per_frame"])

    def test_a_dead_latent_is_undefined_not_zero(self):
        """One distinct code means no path exists. None, not 0.0 (SPEC V29)."""
        from tools.planner.latent_geometry import temporal_fidelity

        dead = np.zeros((40, 64), dtype=np.int8)
        r = temporal_fidelity(dead, ks=(2, 4))
        self.assertIsNone(r["steps_per_frame"])

    def test_a_clip_shorter_than_the_separation_is_skipped(self):
        from tools.planner.latent_geometry import temporal_fidelity

        r = temporal_fidelity(_one_bit_per_frame(n=5), ks=(2, 20))
        self.assertIn(2, r["per_k"])
        self.assertNotIn(20, r["per_k"])

    def test_it_needs_no_planner_and_no_ground_truth(self):
        import inspect
        from tools.planner.latent_geometry import temporal_fidelity

        args = inspect.getfullargspec(temporal_fidelity).args
        for forbidden in ("gt_boxes", "planner", "boxes"):
            self.assertNotIn(forbidden, args)


if __name__ == "__main__":
    unittest.main()
