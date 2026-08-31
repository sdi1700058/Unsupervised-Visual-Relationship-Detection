#!/usr/bin/env python3
"""The E1 summary must not compare a whole arm against another arm's easiest 7%.

The regression these tests lock down is real and it was shipped. On the run of
2026-08-31 the structured arm solved 160 of 160 windows and the unstructured
arm solved 8 of 116. The summariser took the median `bbox_mse` of each, found
unstructured lower, and wrote *"structure does not predict plannability"*. Those
8 windows are the only ones that arm could reach, so they are its easiest, and
they sit on different clips with a linear baseline 3.8x smaller.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.planner import e1_summary  # noqa: E402


def row(reach="True", ratio="1.0", mse="10.0", base="10.0", moving="8",
        iou="0.5", beats="False"):
    return {"reachability": reach, "mse_ratio": ratio, "bbox_mse": mse,
            "baseline_mse": base, "moving_gt_steps": moving, "bbox_iou": iou,
            "beats_baseline": beats}


def arm(solved, total, ratio, mse=None):
    """A summarised arm, built directly rather than through rows."""
    return {"windows": total, "solved": solved, "scored": solved,
            "solve_rate": float(solved) / total, "ratio": ratio,
            "mse": mse if mse is not None else (ratio * 10 if ratio else None),
            "base": 10.0, "iou": 0.5, "beats": 0}


class TestSolveRateLeads(unittest.TestCase):

    def test_the_shipped_regression(self):
        """160/160 against 8/116 must not read as a loss for structured."""
        a = arm(160, 160, 7.426, mse=927.64)
        b = arm(8, 116, 10.226, mse=330.00)
        verdict, caveats = e1_summary.reading(a, b)
        self.assertIn("structure predicts plannability", verdict)
        self.assertNotIn("does not predict", verdict)

    def test_the_selection_bias_is_stated(self):
        """An arm that reached 7% of its windows must be flagged as biased."""
        a = arm(160, 160, 7.426)
        b = arm(8, 116, 10.226)
        _, caveats = e1_summary.reading(a, b)
        joined = " ".join(caveats)
        self.assertIn("optimistically biased", joined)
        self.assertIn("unstructured", joined)

    def test_a_reachability_win_inside_the_confound_is_inconclusive(self):
        """A 1.2x solve-rate gain is under the 1.38x volume advantage."""
        a = arm(60, 100, 5.0)
        b = arm(50, 100, 5.0)
        verdict, _ = e1_summary.reading(a, b)
        self.assertIn("inconclusive", verdict)

    def test_unstructured_reaching_more_is_reported_honestly(self):
        a = arm(40, 100, 5.0)
        b = arm(80, 100, 5.0)
        verdict, _ = e1_summary.reading(a, b)
        self.assertIn("does not predict", verdict)


class TestErrorIsComparedAsARatio(unittest.TestCase):

    def test_raw_error_never_decides_the_verdict(self):
        """Structured with far worse RAW error still wins on the rate."""
        a = arm(100, 100, 5.0, mse=900.0)
        b = arm(10, 100, 5.0, mse=30.0)
        verdict, _ = e1_summary.reading(a, b)
        self.assertIn("structure predicts plannability", verdict)

    def test_ratio_margin_inside_the_confound_is_named_as_such(self):
        a = arm(100, 100, 7.426)
        b = arm(50, 100, 10.226)          # 1.377x, just under 1.38
        _, caveats = e1_summary.reading(a, b)
        joined = " ".join(caveats)
        self.assertIn("does **not** clear", joined)

    def test_ratio_margin_clearing_the_confound_is_credited(self):
        a = arm(100, 100, 5.0)
        b = arm(60, 100, 10.0)            # 2.0x
        _, caveats = e1_summary.reading(a, b)
        self.assertIn("clears the volume confound", " ".join(caveats))


class TestRowSummary(unittest.TestCase):

    def test_reachability_casing_does_not_lose_rows(self):
        """The writer emits 'True' and 'false'; both must parse."""
        rows = [row(reach="True"), row(reach="false"), row(reach="TRUE")]
        out = e1_summary.summarise_rows(rows)
        self.assertEqual(out["solved"], 2)
        self.assertEqual(out["windows"], 3)

    def test_still_windows_are_excluded_from_the_error(self):
        rows = [row(moving="0"), row(moving="8")]
        out = e1_summary.summarise_rows(rows)
        self.assertEqual(out["solved"], 2)
        self.assertEqual(out["scored"], 1)

    def test_no_scoreable_window_is_not_a_verdict(self):
        a = arm(0, 100, None)
        a["scored"] = 0
        b = arm(50, 100, 5.0)
        verdict, _ = e1_summary.reading(a, b)
        self.assertIn("no scorable windows", verdict)

    def test_a_missing_arm_produces_no_reading(self):
        verdict, _ = e1_summary.reading(None, arm(50, 100, 5.0))
        self.assertIn("No reading", verdict)


class TestRender(unittest.TestCase):

    def test_raw_error_row_is_marked_not_comparable(self):
        text = e1_summary.render(arm(160, 160, 7.4, mse=927.6),
                                 arm(8, 116, 10.2, mse=330.0))
        self.assertIn("not comparable across arms", text)
        self.assertIn("median `mse_ratio`", text)

    def test_render_survives_a_missing_arm(self):
        text = e1_summary.render(None, None)
        self.assertIn("Nothing to compare", text)


if __name__ == "__main__":
    unittest.main()
