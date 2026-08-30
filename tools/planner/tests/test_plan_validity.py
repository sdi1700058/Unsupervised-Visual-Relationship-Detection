#!/usr/bin/env python3
"""Tests for M3, plan validity without ground truth.

Synthetic trajectories where the right answer is known by construction: a
smooth walk is admissible, a teleporting one is not, and the thresholds come
from observed motion rather than from a constant somebody chose.

    python3 -m unittest tools/planner/tests/test_plan_validity.py
"""

import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _walk(n=60, step=4.0, size=20.0, seed=0, start=20.0):
    """A single object drifting smoothly, shape (n, 1, 4)."""
    rng = np.random.RandomState(seed)
    x = start + np.cumsum(rng.uniform(-step, step, size=n))
    y = start + np.cumsum(rng.uniform(-step, step, size=n))
    b = np.zeros((n, 1, 4))
    b[:, 0, 0] = x
    b[:, 0, 1] = y
    b[:, 0, 2] = x + size
    b[:, 0, 3] = y + size
    return b


class TestMotionModel(unittest.TestCase):

    def test_bounds_come_from_the_data_not_a_constant(self):
        from tools.planner.plan_validity import motion_model

        slow = motion_model(_walk(step=2.0, seed=1))
        fast = motion_model(_walk(step=40.0, seed=1))
        self.assertLess(slow["max_step"], fast["max_step"])

    def test_absent_frames_do_not_enter_the_motion_model(self):
        """A present-to-absent step is not a displacement (SPEC V30)."""
        from tools.planner.plan_validity import motion_model

        b = _walk(step=3.0, seed=2)
        clean = motion_model(b)
        b2 = b.copy()
        b2[30:35] = 0.0                     # object leaves for five frames
        gapped = motion_model(b2)
        self.assertLess(abs(gapped["max_step"] - clean["max_step"]),
                        clean["max_step"] * 0.5)

    def test_a_model_needs_some_motion(self):
        from tools.planner.plan_validity import motion_model

        still = np.tile(np.array([[[10.0, 10.0, 30.0, 30.0]]]), (20, 1, 1))
        self.assertIsNone(motion_model(still)["max_step"])


class TestValidity(unittest.TestCase):

    def test_a_real_trajectory_is_admissible_against_its_own_model(self):
        from tools.planner.plan_validity import motion_model, plan_validity

        b = _walk(n=120, step=3.0, seed=3)
        m = motion_model(b[:80])
        r = plan_validity(b[80:], m)
        self.assertGreater(r["validity"], 0.95)

    def test_teleporting_is_caught(self):
        from tools.planner.plan_validity import motion_model, plan_validity

        b = _walk(n=120, step=3.0, seed=4)
        m = motion_model(b[:80])

        clean = plan_validity(b[80:], m)

        # One jump in a forty-step window is one bad step, so the right
        # assertion is that validity DROPS, not that it falls below some
        # absolute line -- 39 of 40 clean is 0.975 and that is correct.
        bad = b[80:].copy()
        bad[5:, 0, 0] += 400.0              # jump across the canvas
        bad[5:, 0, 2] += 400.0
        r = plan_validity(bad, m)
        self.assertLess(r["validity"], clean["validity"])
        self.assertGreater(r["teleport_rate"], 0.0)

        # Every step a jump: now it should collapse.
        rng = np.random.RandomState(11)
        worse = b[80:].copy()
        worse[:, 0, 0] += rng.choice([-400.0, 400.0], size=len(worse))
        worse[:, 0, 2] += worse[:, 0, 0] - b[80:, 0, 0]
        self.assertLess(plan_validity(worse, m)["validity"], 0.5)

    def test_an_object_blinking_in_and_out_is_caught(self):
        from tools.planner.plan_validity import motion_model, plan_validity

        b = _walk(n=120, step=3.0, seed=5)
        m = motion_model(b[:80])

        bad = b[80:].copy()
        bad[1::2] = 0.0                     # present, absent, present, absent
        r = plan_validity(bad, m)
        self.assertGreater(r["flicker_rate"], 0.5)

    def test_a_box_that_inverts_is_caught(self):
        from tools.planner.plan_validity import motion_model, plan_validity

        b = _walk(n=120, step=3.0, seed=6)
        m = motion_model(b[:80])

        bad = b[80:].copy()
        bad[:, 0, 2] = bad[:, 0, 0] - 5.0   # x2 < x1
        r = plan_validity(bad, m)
        self.assertGreater(r["malformed_rate"], 0.9)

    def test_leaving_the_canvas_is_caught(self):
        from tools.planner.plan_validity import motion_model, plan_validity

        b = _walk(n=120, step=3.0, seed=7)
        m = motion_model(b[:80])

        bad = b[80:].copy()
        bad[:, 0, 0] -= 900.0
        bad[:, 0, 2] -= 900.0
        r = plan_validity(bad, m, width=300, height=200)
        self.assertGreater(r["offcanvas_rate"], 0.9)

    def test_validity_needs_no_ground_truth(self):
        """The whole point: the signature takes no gt_boxes."""
        import inspect
        from tools.planner.plan_validity import plan_validity

        args = inspect.getfullargspec(plan_validity).args
        for forbidden in ("gt_boxes", "gt", "truth", "reference"):
            self.assertNotIn(forbidden, args)


class TestDiscrimination(unittest.TestCase):

    def test_real_trajectories_score_above_scrambled_ones(self):
        """The check that makes M3 worth reporting.

        A validity score is only useful if it separates a plausible
        trajectory from an implausible one. If a scrambled trajectory scores
        the same, the measure is not measuring anything.
        """
        from tools.planner.plan_validity import (motion_model, plan_validity,
                                                 discrimination)

        b = _walk(n=200, step=3.0, seed=8)
        m = motion_model(b[:120])
        real = b[120:]

        rng = np.random.RandomState(0)
        scrambled = real[rng.permutation(len(real))]

        r = plan_validity(real, m)["validity"]
        s = plan_validity(scrambled, m)["validity"]
        self.assertGreater(r, s)

        d = discrimination(real, scrambled, m)
        self.assertGreater(d["separation"], 0.1)


if __name__ == "__main__":
    unittest.main()
