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

from tools.planner import plan_validity  # noqa: E402


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



class TestBoundCalibration(unittest.TestCase):
    """The default bound, and the guard that stops it being tightened further.

    Measured 2026-08-30 across 20 VidVRD and 13 Something-Else clips. Tighter
    bounds always raise separation, so separation alone would drive the
    threshold to zero. What stops it is that a REAL trajectory must keep
    scoring high validity:

        pct  slack   VidVRD real-validity / sep    SomethingElse
        99   1.5     1.000 / 0.147                 1.000 / 0.000
        95   1.0     1.000 / 0.269                 1.000 / 0.083   <- default
        90   1.0     0.990 / 0.288                 1.000 / 0.095
        75   1.0     0.922 / 0.378  REAL FLAGGED   0.974 / 0.154

    99/1.5 was the original default and it is too loose: it left the measure
    silent on 8 of 13 Something-Else clips.
    """

    def _walk_corpus(self, n=200, step=3.0, seed=0, bin_px=5.0):
        """A walk that stays on the canvas AND lies on bin edges.

        Both details are needed for this to be a fair model of what M3 sees,
        and each was established by measurement after the first version of
        this test failed:

        1. An unclamped 200-step walk drifts off a 300x200 canvas, so
           `offcanvas_rate` decides the test rather than the displacement
           bound.

        2. **Quantisation is the important one.** Every trajectory M3 ever
           scores comes out of `latents_to_boxes`, so it lies on bin edges --
           300/60 = 5 px. A continuous walk has no zero-length steps, and a
           95th-percentile bound then flags about 5% of its steps BY
           DEFINITION. A quantised walk spends about half its steps not
           changing bin at all, and a zero step can never exceed the bound.
           Measured: 0.058 of real steps flagged continuous, **0.000**
           quantised.

        A hypothesis was tested and falsified on the way here: that
        autocorrelation ("momentum") explained the gap. It does not -- the
        flagged fraction stays at 0.06-0.07 for every autocorrelation from 0.0
        to 0.9. Quantisation is the whole of it.
        """
        b = _walk(n=n, step=step, seed=seed, start=120.0)
        x = np.clip(b[:, 0, 0], 5.0, 250.0)
        y = np.clip(b[:, 0, 1], 5.0, 160.0)
        x = np.floor(x / bin_px) * bin_px
        y = np.floor(y / bin_px) * bin_px
        b[:, 0, 0], b[:, 0, 1] = x, y
        b[:, 0, 2], b[:, 0, 3] = x + 20.0, y + 20.0
        return b

    def test_the_default_does_not_flag_a_real_trajectory(self):
        from tools.planner.plan_validity import motion_model, plan_validity

        for seed in range(6):
            b = self._walk_corpus(seed=seed)
            m = motion_model(b[:140])
            v = plan_validity(b[140:], m, width=300, height=200)["validity"]
            self.assertGreaterEqual(v, 0.95,
                                    "default bound flags real motion (seed %d)" % seed)

    def test_the_default_still_separates_real_from_scrambled(self):
        from tools.planner.plan_validity import motion_model, discrimination

        rng = np.random.RandomState(0)
        seps = []
        for seed in range(6):
            b = self._walk_corpus(seed=seed)
            m = motion_model(b[:140])
            real = b[140:]
            seps.append(discrimination(real, real[rng.permutation(len(real))],
                                       m, width=300, height=200)["separation"])
        self.assertGreater(float(np.median(seps)), 0.05)

    def test_a_looser_bound_separates_less(self):
        """Which is why the default moved off 99/1.5."""
        from tools.planner.plan_validity import motion_model, discrimination

        b = self._walk_corpus(seed=3)
        real = b[140:]
        rng = np.random.RandomState(1)
        scrambled = real[rng.permutation(len(real))]

        tight = discrimination(real, scrambled, motion_model(b[:140], percentile=95.0),
                               width=300, height=200, slack=1.0)["separation"]
        loose = discrimination(real, scrambled, motion_model(b[:140], percentile=99.0),
                               width=300, height=200, slack=1.5)["separation"]
        self.assertGreaterEqual(tight, loose)

    def test_percentile_is_settable(self):
        from tools.planner.plan_validity import motion_model

        b = self._walk_corpus(seed=2)
        self.assertLess(motion_model(b, percentile=75.0)["max_step"],
                        motion_model(b, percentile=99.0)["max_step"])

if __name__ == "__main__":
    unittest.main()


class TestPaddingIsNotAnAdmissibleStep(unittest.TestCase):
    """An empty slot cannot fail, so counting it inflates the score.

    `validity` divided by `(n_frames - 1) * n_objs`, which includes the
    all-zero padding slots the bake writes to fill `--max-objects`. A padding
    slot is absent at both ends of every step, so it never teleports, never
    flickers, is never malformed and is never off-canvas. It was therefore
    counted as an admissible step, every time.

    Three slots holding one real object gave `validity >= 0.667` before the
    planner did anything at all. The scrambled control receives the same lift,
    so `separation` was never affected -- but the ADMISSIBLE threshold of 0.95
    is an absolute reading of `validity`, and 0.972 is quoted in the supervisor
    report.
    """

    def _one_object(self, n_frames, n_slots, step=2.0):
        """One object moving gently; the remaining slots are padding."""
        b = np.zeros((n_frames, n_slots, 4))
        for t in range(n_frames):
            x = 10.0 + step * t
            b[t, 0] = [x, 10.0, x + 5.0, 15.0]
        return b

    def test_padding_slots_do_not_raise_the_score(self):
        """The bug, stated as the number it produced."""
        model = {"max_step": 50.0}
        tight = plan_validity.plan_validity(self._one_object(5, 1), model)
        padded = plan_validity.plan_validity(self._one_object(5, 3), model)
        self.assertAlmostEqual(padded["validity"], tight["validity"], places=9)

    def test_a_padded_trace_cannot_beat_an_unpadded_one(self):
        """More empty slots must never look like a better plan."""
        model = {"max_step": 50.0}
        scores = [plan_validity.plan_validity(self._one_object(5, n),
                                         model)["validity"]
                  for n in (1, 2, 5, 10)]
        self.assertEqual(len(set(round(s, 9) for s in scores)), 1)

    def test_one_bad_step_is_not_diluted_by_padding(self):
        """The consequence: padding hid real failures in proportion to itself."""
        model = {"max_step": 5.0}
        b = self._one_object(3, 4)
        b[2, 0] = [500.0, 10.0, 505.0, 15.0]        # a teleport
        got = plan_validity.plan_validity(b, model)
        self.assertAlmostEqual(got["validity"], 0.5, places=9)

    def test_the_denominator_is_reported(self):
        """So a reader can see what the fraction was taken over."""
        got = plan_validity.plan_validity(self._one_object(5, 4),
                                     {"max_step": 50.0})
        self.assertEqual(got["n_steps_scored"], 4)
        self.assertEqual(got["n_objects"], 4)

    def test_the_rates_use_the_same_denominator(self):
        model = {"max_step": 5.0}
        b = self._one_object(3, 4)
        b[2, 0] = [500.0, 10.0, 505.0, 15.0]
        got = plan_validity.plan_validity(b, model)
        self.assertAlmostEqual(got["teleport_rate"], 0.5, places=9)

    def test_an_all_padding_trace_scores_none_rather_than_one(self):
        """Nothing to judge is not the same as everything being right."""
        got = plan_validity.plan_validity(np.zeros((5, 3, 4)), {"max_step": 50.0})
        self.assertIsNone(got["validity"])

    def test_an_object_appearing_midway_still_counts(self):
        """A slot present at one end of a step is a real step."""
        b = np.zeros((3, 2, 4))
        b[1, 0] = [10.0, 10.0, 15.0, 15.0]
        b[2, 0] = [12.0, 10.0, 17.0, 15.0]
        got = plan_validity.plan_validity(b, {"max_step": 50.0})
        self.assertEqual(got["n_steps_scored"], 2)


class TestNothingScoredIsReportedAsSuch(unittest.TestCase):
    """`validity` can now be None, and everything downstream must survive it.

    Before padding was excluded, `validity` was always a float, because an
    all-padding trace scored 1.0. Now a trace with nothing to judge returns
    None, and `verdict` and `discrimination` have to say so rather than raise
    or invent a comparison.
    """

    EMPTY = staticmethod(lambda: np.zeros((5, 3, 4)))

    def test_discrimination_reports_no_separation_rather_than_raising(self):
        d = plan_validity.discrimination(self.EMPTY(), self.EMPTY(),
                                         {"max_step": 50.0})
        self.assertIsNone(d["separation"])

    def test_verdict_on_an_unscoreable_trace_is_silent(self):
        result = plan_validity.plan_validity(self.EMPTY(), {"max_step": 50.0})
        result["discrimination"] = plan_validity.discrimination(
            self.EMPTY(), self.EMPTY(), {"max_step": 50.0})
        text = plan_validity.verdict(result)
        self.assertTrue(text.startswith("SILENT"))

    def test_verdict_names_why_it_is_silent(self):
        result = plan_validity.plan_validity(self.EMPTY(), {"max_step": 50.0})
        result["discrimination"] = plan_validity.discrimination(
            self.EMPTY(), self.EMPTY(), {"max_step": 50.0})
        self.assertIn("no object", plan_validity.verdict(result).lower())

    def test_a_real_trace_still_gets_a_real_verdict(self):
        """The guard must not swallow the ordinary case."""
        b = _walk(n=120, step=3.0, seed=21)
        m = plan_validity.motion_model(b[:80])
        result = plan_validity.plan_validity(b[80:], m)
        rng = np.random.RandomState(0)
        result["discrimination"] = plan_validity.discrimination(
            b[80:], b[80:][rng.permutation(40)], m)
        self.assertFalse(plan_validity.verdict(result).startswith("SILENT"))
