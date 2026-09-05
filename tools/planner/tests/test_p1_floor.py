#!/usr/bin/env python3
"""Tests for the summariser that fills in the floor columns after the fact.

Everything here is synthetic. The point of the file is that the ways this
particular summariser can be wrong are all silent: it can divide by the wrong
floor, score against an export that has since been rebuilt, or let a clip with
four windows outvote a clip with two. None of those raise.

    python3 -m unittest tools/planner/tests/test_p1_floor.py
"""

import os
import sys
import tempfile
import unittest
import xml.dom.minidom
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def _boxes(coords):
    """(T, 1, 4) boxes from a list of (x1, y1, x2, y2)."""
    return np.array([[c] for c in coords], dtype=np.float64)


class TestColumnPlacement(unittest.TestCase):

    def test_the_new_columns_land_where_the_run_script_puts_them(self):
        from tools.planner.p1_floor import insert_columns

        header = ["temporal_order", "decode_fallbacks", "wall_s"]
        out = insert_columns(header, "temporal_order",
                             ("floor_ratio", "quantisation_floor"))
        self.assertEqual(out, ["temporal_order", "floor_ratio",
                               "quantisation_floor", "decode_fallbacks",
                               "wall_s"])

    def test_running_twice_does_not_duplicate_a_column(self):
        from tools.planner.p1_floor import insert_columns

        header = ["temporal_order", "floor_ratio", "quantisation_floor",
                  "wall_s"]
        self.assertEqual(
            insert_columns(header, "temporal_order",
                           ("floor_ratio", "quantisation_floor")),
            header)


class TestFloorDefinitions(unittest.TestCase):

    def test_the_window_floor_ignores_the_endpoints(self):
        """The window is scored on its k-2 middle frames, so the floor is too.

        The endpoints here sit exactly on a bin edge and cost nothing, while
        the middles do not. A floor that swept the endpoints in would be
        diluted by two free frames and would flatter every ratio built on it.
        """
        from tools.planner.p1_floor import window_floor, clip_floor

        boxes = _boxes([(0.0, 0.0, 5.0, 5.0),     # frame 0, on the grid
                        (2.5, 2.5, 7.5, 7.5),     # frame 1, off the grid
                        (2.5, 2.5, 7.5, 7.5),     # frame 2, off the grid
                        (0.0, 0.0, 5.0, 5.0)])    # frame 3, on the grid
        middles = window_floor(boxes, 0, 3)
        whole = clip_floor(boxes)
        self.assertGreater(middles, whole)

    def test_the_two_floors_are_not_the_same_number(self):
        """Pinned because the published pair was stated in the second one.

        A reader who assumes `clip_floor_ratio` and `floor_ratio` are the same
        quantity will reconcile two tables that were never comparable.
        """
        from tools.planner.p1_floor import window_floor, clip_floor

        boxes = _boxes([(0.0, 0.0, 5.0, 5.0)] * 4
                       + [(2.5, 2.5, 7.5, 7.5)] * 4)
        self.assertNotAlmostEqual(window_floor(boxes, 0, 3), clip_floor(boxes))

    def test_a_window_with_no_present_object_has_no_floor(self):
        from tools.planner.p1_floor import window_floor

        boxes = np.zeros((4, 1, 4), dtype=np.float64)
        self.assertIsNone(window_floor(boxes, 0, 3))


class TestExportFingerprint(unittest.TestCase):
    """The export must be shown to be the one the run was scored against."""

    def setUp(self):
        self.boxes = _boxes([(10.0, 10.0, 40.0, 40.0),
                             (20.0, 15.0, 50.0, 45.0),
                             (30.0, 20.0, 60.0, 50.0),
                             (55.0, 25.0, 95.0, 55.0)])

    def test_the_recomputed_baseline_reproduces_a_matching_export(self):
        from tools.planner.p1_floor import recompute_baseline, agrees

        value = recompute_baseline(self.boxes, 0, 3)
        self.assertIsNotNone(value)
        self.assertTrue(agrees(value, repr(value)))

    def test_a_rebuilt_export_is_rejected_rather_than_scored(self):
        from tools.planner.p1_floor import recompute_baseline, agrees

        recorded = recompute_baseline(self.boxes, 0, 3)
        moved = self.boxes.copy()
        moved[2, 0] += 4.0
        self.assertFalse(agrees(recompute_baseline(moved, 0, 3), repr(recorded)))

    def test_float32_rounding_is_not_treated_as_a_different_export(self):
        from tools.planner.p1_floor import recompute_baseline, agrees

        exact = recompute_baseline(self.boxes, 0, 3)
        rounded = recompute_baseline(self.boxes.astype(np.float32).astype(
            np.float64), 0, 3)
        self.assertTrue(agrees(rounded, repr(exact)))

    def test_a_window_with_nothing_between_the_endpoints_has_no_baseline(self):
        from tools.planner.p1_floor import recompute_baseline

        self.assertIsNone(recompute_baseline(self.boxes, 0, 1))


class TestAggregation(unittest.TestCase):

    def test_a_clip_with_more_windows_does_not_get_more_votes(self):
        """Rows are not clips. Counting rows is how a margin gets invented."""
        from tools.planner.p1_floor import per_clip_medians

        runs = {
            "busy": [{"floor_ratio": "100.0"}] * 9,
            "quiet": [{"floor_ratio": "1.0"}],
            "other": [{"floor_ratio": "2.0"}],
        }
        medians = per_clip_medians(runs, "floor_ratio")
        self.assertEqual(len(medians), 3)
        self.assertAlmostEqual(medians["busy"], 100.0)
        self.assertAlmostEqual(medians["quiet"], 1.0)

    def test_a_clip_with_no_scored_window_is_absent_not_zero(self):
        from tools.planner.p1_floor import per_clip_medians

        runs = {"empty": [{"floor_ratio": ""}], "ok": [{"floor_ratio": "3.0"}]}
        self.assertEqual(sorted(per_clip_medians(runs, "floor_ratio")), ["ok"])

    def test_pairing_uses_only_the_clips_both_arms_scored(self):
        """The count must not compare one arm's whole set against a subset."""
        from tools.planner.p1_floor import paired

        a = {"x": 1.0, "y": 2.0, "z": 3.0}
        b = {"x": 5.0, "y": 1.0}
        clips, closer = paired(a, b)
        self.assertEqual(clips, ["x", "y"])
        self.assertEqual(closer, 1)

    def test_the_shared_denominator_is_the_reference_arms_floor(self):
        from tools.planner.p1_floor import medians_against

        runs = {"clip": [{"bbox_mse": "40.0"}, {"bbox_mse": "20.0"}]}
        self.assertAlmostEqual(medians_against(runs, {"clip": 10.0})["clip"],
                               3.0)

    def test_a_clip_with_no_shared_floor_is_dropped(self):
        from tools.planner.p1_floor import medians_against

        self.assertEqual(medians_against({"c": [{"bbox_mse": "1.0"}]}, {}), {})


class TestSummaryRewrite(unittest.TestCase):

    def _run(self, tmp, rows):
        run_dir = os.path.join(tmp, "p3-clipA")
        os.makedirs(os.path.join(run_dir, "logs"))
        header = ("export,method,init,goal,bbox_mse,baseline_mse,"
                  "temporal_order,decode_fallbacks,wall_s")
        with open(os.path.join(run_dir, "summary.csv"), "w") as handle:
            handle.write(header + "\n")
            for row in rows:
                handle.write(row + "\n")
        return run_dir

    def test_a_summary_with_two_rows_for_one_window_is_refused(self):
        """80 windows scored twice are 80 windows, not 160."""
        from tools.planner.p1_floor import score_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run(tmp, ["clipA,pddl,0,3,1,1,1,0,1",
                                      "clipA,pddl,0,3,2,2,1,0,1"])
            with self.assertRaises(ValueError):
                score_run(run_dir, None, {})

    def test_a_row_without_an_export_gets_no_floor_and_is_counted(self):
        from tools.planner.p1_floor import score_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run(tmp, ["clipA,pddl,0,3,7.5,2.0,1,0,1"])
            header, rows, report = score_run(run_dir, os.path.join(tmp, "gone"),
                                             {})
            self.assertIn("floor_ratio", header)
            self.assertEqual(rows[0].get("floor_ratio", ""), "")
            self.assertEqual(len(report["no_export"]), 1)
            self.assertEqual(report["filled"], 0)

    def test_an_unscored_row_is_not_counted_as_a_window(self):
        from tools.planner.p1_floor import score_run

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run(tmp, ["clipA,pddl,0,3,,,,,"])
            _, _, report = score_run(run_dir, None, {})
            self.assertEqual(report["rows"], 1)
            self.assertEqual(report["scored"], 0)

    def test_the_floor_columns_are_filled_from_a_matching_export(self):
        from tools.planner.p1_floor import (recompute_baseline, score_run,
                                            window_floor, clip_floor)

        boxes = _boxes([(10.0, 10.0, 40.0, 40.0),
                        (23.0, 17.0, 53.0, 47.0),
                        (31.0, 21.0, 61.0, 51.0),
                        (55.0, 25.0, 95.0, 55.0)])
        with tempfile.TemporaryDirectory() as tmp:
            exports = os.path.join(tmp, "exports")
            os.makedirs(exports)
            np.savez(os.path.join(exports, "clipA.npz"), gt_boxes=boxes)
            base = recompute_baseline(boxes, 0, 3)
            run_dir = self._run(tmp, ["clipA,pddl,0,3,60.0,%r,1,0,1" % base])

            _, rows, report = score_run(run_dir, exports, {})
            self.assertEqual(report["filled"], 1)
            self.assertEqual(report["unverified"], [])
            self.assertAlmostEqual(float(rows[0]["quantisation_floor"]),
                                   window_floor(boxes, 0, 3))
            self.assertAlmostEqual(float(rows[0]["floor_ratio"]),
                                   60.0 / window_floor(boxes, 0, 3))
            self.assertAlmostEqual(float(rows[0]["clip_floor"]),
                                   clip_floor(boxes))

    def test_a_floor_the_harness_already_wrote_is_confirmed_not_replaced(self):
        """Overwriting would hide a disagreement instead of reporting one."""
        from tools.planner.p1_floor import (recompute_baseline, score_run,
                                            window_floor)

        boxes = _boxes([(10.0, 10.0, 40.0, 40.0),
                        (23.0, 17.0, 53.0, 47.0),
                        (31.0, 21.0, 61.0, 51.0),
                        (55.0, 25.0, 95.0, 55.0)])
        with tempfile.TemporaryDirectory() as tmp:
            exports = os.path.join(tmp, "exports")
            os.makedirs(exports)
            np.savez(os.path.join(exports, "clipA.npz"), gt_boxes=boxes)
            run_dir = os.path.join(tmp, "p3-clipA")
            os.makedirs(os.path.join(run_dir, "logs"))
            base = recompute_baseline(boxes, 0, 3)
            wrong = repr(60.0 / window_floor(boxes, 0, 3) * 2.0)
            with open(os.path.join(run_dir, "summary.csv"), "w") as handle:
                handle.write("export,method,init,goal,bbox_mse,baseline_mse,"
                             "temporal_order,floor_ratio,quantisation_floor,"
                             "wall_s\n")
                handle.write("clipA,pddl,0,3,60.0,%r,1,%s,1.0,1\n"
                             % (base, wrong))

            _, rows, report = score_run(run_dir, exports, {})
            self.assertEqual(report["filled"], 0)
            self.assertEqual(report["confirmed"], 0)
            self.assertEqual(len(report["disagreed"]), 1)
            self.assertEqual(rows[0]["floor_ratio"], wrong)

    def test_an_agreeing_floor_is_counted_as_confirmation(self):
        from tools.planner.p1_floor import (recompute_baseline, score_run,
                                            window_floor)

        boxes = _boxes([(10.0, 10.0, 40.0, 40.0),
                        (23.0, 17.0, 53.0, 47.0),
                        (31.0, 21.0, 61.0, 51.0),
                        (55.0, 25.0, 95.0, 55.0)])
        with tempfile.TemporaryDirectory() as tmp:
            exports = os.path.join(tmp, "exports")
            os.makedirs(exports)
            np.savez(os.path.join(exports, "clipA.npz"), gt_boxes=boxes)
            run_dir = os.path.join(tmp, "p3-clipA")
            os.makedirs(os.path.join(run_dir, "logs"))
            base = recompute_baseline(boxes, 0, 3)
            floor = window_floor(boxes, 0, 3)
            with open(os.path.join(run_dir, "summary.csv"), "w") as handle:
                handle.write("export,method,init,goal,bbox_mse,baseline_mse,"
                             "temporal_order,floor_ratio,quantisation_floor,"
                             "wall_s\n")
                handle.write("clipA,pddl,0,3,60.0,%r,1,%r,%r,1\n"
                             % (base, 60.0 / floor, floor))

            _, _, report = score_run(run_dir, exports, {})
            self.assertEqual(report["confirmed"], 1)
            self.assertEqual(report["disagreed"], [])

    def test_a_stale_export_leaves_the_columns_empty(self):
        from tools.planner.p1_floor import score_run

        boxes = _boxes([(10.0, 10.0, 40.0, 40.0),
                        (23.0, 17.0, 53.0, 47.0),
                        (31.0, 21.0, 61.0, 51.0),
                        (55.0, 25.0, 95.0, 55.0)])
        with tempfile.TemporaryDirectory() as tmp:
            exports = os.path.join(tmp, "exports")
            os.makedirs(exports)
            np.savez(os.path.join(exports, "clipA.npz"), gt_boxes=boxes)
            run_dir = self._run(tmp, ["clipA,pddl,0,3,60.0,999.0,1,0,1"])

            _, rows, report = score_run(run_dir, exports, {})
            self.assertEqual(report["filled"], 0)
            self.assertEqual(len(report["unverified"]), 1)
            self.assertEqual(rows[0].get("floor_ratio", ""), "")

    def test_the_export_recorded_in_the_log_wins_over_the_directory(self):
        from tools.planner.p1_floor import export_path_from_log

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run(tmp, ["clipA,pddl,0,3,1,1,1,0,1"])
            with open(os.path.join(run_dir, "logs",
                                   "clipA_pddl_0_3.log"), "w") as handle:
                handle.write("method   pddl\nexport   /elsewhere/clipA.npz\n")
            self.assertEqual(
                export_path_from_log(run_dir, {"export": "clipA",
                                               "method": "pddl",
                                               "init": "0", "goal": "3"}),
                "/elsewhere/clipA.npz")

    def test_a_missing_log_is_reported_as_absent_rather_than_guessed(self):
        from tools.planner.p1_floor import export_path_from_log

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run(tmp, ["clipA,pddl,0,3,1,1,1,0,1"])
            self.assertIsNone(
                export_path_from_log(run_dir, {"export": "clipA",
                                               "method": "pddl",
                                               "init": "0", "goal": "3"}))


class TestFigure(unittest.TestCase):

    def test_the_figure_is_well_formed_xml(self):
        """This project has shipped an unopenable SVG three times."""
        from tools.planner.p1_floor import figure

        clips = ["a", "b", "c"]
        vals = dict(zip(clips, [1.2, 2.0, 5.0]))
        other = dict(zip(clips, [10.0, 90.0, 300.0]))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.svg")
            figure(path, clips, vals, other, vals, other, "floor 1 to 2")
            xml.dom.minidom.parse(path)

    def test_no_raw_angle_bracket_reaches_the_text(self):
        from tools.planner.p1_floor import figure

        clips = ["a"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "f.svg")
            figure(path, clips, {"a": 1.0}, {"a": 9.0}, {"a": 1.0},
                   {"a": 9.0}, "title")
            with open(path) as handle:
                body = handle.read()
        for chunk in body.split("<")[1:]:
            self.assertIn(">", chunk)


if __name__ == "__main__":
    unittest.main()
