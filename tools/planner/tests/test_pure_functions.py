#!/usr/bin/env python3
"""Tests for the planner code that runs without keras.

Everything here works on synthetic latents, so the suite runs on a laptop
with no trained model, no GPU and no lisp toolchain. Anything that needs a
real model belongs in a smoke run, not here.

    python3 -m unittest tools/planner/tests/test_pure_functions.py
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class TestWindows(unittest.TestCase):

    def test_window_covers_the_frames_between_the_endpoints(self):
        from tools.planner.common.windows import make_windows

        windows = make_windows(n_frames=10, k=5, stride=1)
        self.assertEqual(len(windows), 6)

        first = windows[0]
        self.assertEqual(first["init"], 0)
        self.assertEqual(first["goal"], 4)
        self.assertEqual(first["intermediate"], [1, 2, 3])
        self.assertEqual(first["plan_length"], 4)

    def test_stride_thins_the_windows(self):
        from tools.planner.common.windows import make_windows

        dense = make_windows(n_frames=20, k=5, stride=1)
        sparse = make_windows(n_frames=20, k=5, stride=4)
        self.assertGreater(len(dense), len(sparse))
        # A window starting at 16 would need frame 20, past the last index.
        self.assertEqual([w["init"] for w in sparse], [0, 4, 8, 12])

    def test_window_smaller_than_three_frames_is_rejected(self):
        from tools.planner.common.windows import make_windows

        # k=2 leaves nothing between the endpoints, so there is nothing to score.
        with self.assertRaises(ValueError):
            make_windows(n_frames=10, k=2)

    def test_trace_of_the_right_length_passes_through_untouched(self):
        from tools.planner.common.windows import extract_intermediate_states

        trace = np.arange(5 * 4).reshape(5, 4)
        mid, exact = extract_intermediate_states(trace, expected_count=3)
        self.assertTrue(exact)
        np.testing.assert_array_equal(mid, trace[1:-1])

    def test_longer_trace_is_resampled_and_flagged(self):
        from tools.planner.common.windows import extract_intermediate_states

        trace = np.arange(9 * 4).reshape(9, 4)
        mid, exact = extract_intermediate_states(trace, expected_count=3)
        self.assertFalse(exact)
        self.assertEqual(len(mid), 3)

    def test_plan_that_skips_straight_to_the_goal_repeats_the_init(self):
        from tools.planner.common.windows import extract_intermediate_states

        trace = np.array([[0, 0], [1, 1]])
        mid, exact = extract_intermediate_states(trace, expected_count=2)
        self.assertFalse(exact)
        self.assertEqual(len(mid), 2)

    def test_linear_baseline_sits_between_the_endpoints(self):
        from tools.planner.common.windows import linear_interp_bboxes

        init = np.array([[0.0, 0.0, 10.0, 10.0]])
        goal = np.array([[100.0, 100.0, 110.0, 110.0]])

        boxes = linear_interp_bboxes(init, goal, n_intermediate=3)
        self.assertEqual(boxes.shape, (3, 1, 4))
        self.assertGreater(boxes[0, 0, 0], init[0, 0])
        self.assertLess(boxes[-1, 0, 0], goal[0, 0])
        self.assertTrue(np.all(np.diff(boxes[:, 0, 0]) > 0))


class TestBfs(unittest.TestCase):

    def test_deltas_are_deduplicated(self):
        from tools.planner.bfs.planner import mine_deltas

        # 0->1 and 2->3 flip the same bit, so that is one delta. 1->2 changes
        # nothing and must be dropped rather than kept as a self-loop.
        z = np.array([[0, 0], [1, 0], [1, 0], [0, 0]], dtype=np.int8)
        deltas = mine_deltas(z)
        self.assertEqual(len(deltas), 1)
        np.testing.assert_array_equal(deltas[0], [1, 0])

    def test_search_walks_one_bit_at_a_time(self):
        from tools.planner.bfs.planner import mine_deltas, search

        z = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 1],
                      [0, 1, 1, 1],
                      [1, 1, 1, 1]], dtype=np.int8)
        deltas = mine_deltas(z)

        found, trace, _ = search(z[0], z[-1], deltas, time_budget_s=5)
        self.assertTrue(found)
        self.assertEqual(len(trace) - 1, 4)
        np.testing.assert_array_equal(trace[0], z[0])
        np.testing.assert_array_equal(trace[-1], z[-1])

    def test_goal_equal_to_init_needs_no_actions(self):
        from tools.planner.bfs.planner import mine_deltas, search

        z = np.array([[0, 0], [1, 0]], dtype=np.int8)
        found, trace, _ = search(z[0], z[0], mine_deltas(z), time_budget_s=5)
        self.assertTrue(found)
        self.assertEqual(len(trace), 1)

    def test_unreachable_goal_reports_failure(self):
        from tools.planner.bfs.planner import search

        # The only delta touches bit 0, so bit 1 can never be set.
        deltas = np.array([[1, 0]], dtype=np.int8)
        found, trace, _ = search(np.array([0, 0], dtype=np.int8),
                                 np.array([0, 1], dtype=np.int8),
                                 deltas, time_budget_s=2)
        self.assertFalse(found)
        self.assertEqual(len(trace), 0)

    def test_repeated_searches_return_the_same_plan(self):
        from tools.planner.bfs.planner import mine_deltas, search

        z = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                     dtype=np.int8)
        deltas = mine_deltas(z)

        first = search(z[0], z[-1], deltas, time_budget_s=5)[1]
        second = search(z[0], z[-1], deltas, time_budget_s=5)[1]
        np.testing.assert_array_equal(first, second)


class TestPddl(unittest.TestCase):

    def setUp(self):
        self.pre = np.array([[0, 1, 0, 0],
                             [1, 1, 0, 0],
                             [0, 1, 0, 0]], dtype=np.int8)
        self.suc = np.array([[1, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 1, 1, 0]], dtype=np.int8)

    def test_actions_csv_round_trip(self):
        from tools.planner.pddl.planner import read_actions_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actions.csv"
            np.savetxt(path, np.concatenate([self.pre, self.suc], axis=1), "%d")

            pre, suc = read_actions_csv(path)
            np.testing.assert_array_equal(pre, self.pre)
            np.testing.assert_array_equal(suc, self.suc)

    def test_odd_width_csv_is_rejected(self):
        from tools.planner.pddl.planner import read_actions_csv

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            np.savetxt(path, np.zeros((2, 5), dtype=int), "%d")
            with self.assertRaises(ValueError):
                read_actions_csv(path)

    def test_effects_split_into_add_and_delete(self):
        from tools.planner.pddl.planner import distinct_effects

        effects = distinct_effects(self.pre, self.suc)
        self.assertEqual(len(effects), 3)
        self.assertEqual(effects[0], {"add": [0], "del": [], "count": 1})
        self.assertEqual(effects[1], {"add": [], "del": [1], "count": 1})
        self.assertEqual(effects[2], {"add": [2], "del": [], "count": 1})

    def test_no_op_transitions_are_dropped(self):
        from tools.planner.pddl.planner import distinct_effects

        same = np.array([[1, 0], [1, 0]], dtype=np.int8)
        self.assertEqual(distinct_effects(same, same), [])

    def test_domain_declares_one_action_per_effect(self):
        from tools.planner.pddl.planner import distinct_effects, write_domain

        effects = distinct_effects(self.pre, self.suc)
        with tempfile.TemporaryDirectory() as tmp:
            text = write_domain(effects, 4, Path(tmp) / "domain.pddl").read_text()

        self.assertIn("(define (domain fosae)", text)
        self.assertIn(":requirements :strips :negative-preconditions", text)
        self.assertEqual(text.count("(:action"), 3)
        self.assertEqual(text.count("("), text.count(")"))

    def test_goal_pins_every_bit(self):
        from tools.planner.pddl.planner import write_problem

        with tempfile.TemporaryDirectory() as tmp:
            text = write_problem(np.array([0, 1, 0, 0]),
                                 np.array([0, 1, 1, 0]),
                                 4, Path(tmp) / "problem.pddl").read_text()

        self.assertIn("(:init (bit_1))", text)
        # Two bits on, two off, so two plain literals and two negated ones.
        self.assertEqual(text.count("(not (bit_"), 2)

    def test_problem_rejects_a_latent_of_the_wrong_width(self):
        from tools.planner.pddl.planner import write_problem

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_problem(np.zeros(3), np.zeros(4), 4,
                              Path(tmp) / "problem.pddl")

    def test_plan_file_parses_with_or_without_parentheses(self):
        from tools.planner.pddl.planner import read_plan

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sas_plan"
            path.write_text("; cost = 2\n(act_0)\nact_5\n")
            self.assertEqual(read_plan(path), [0, 5])

    def test_replay_applies_the_effects_in_order(self):
        from tools.planner.pddl.planner import distinct_effects, replay

        effects = distinct_effects(self.pre, self.suc)
        trace = replay(np.array([0, 1, 0, 0], dtype=np.int8), effects, [2])

        self.assertEqual(len(trace), 2)
        np.testing.assert_array_equal(trace[-1], [0, 1, 1, 0])


class TestMetrics(unittest.TestCase):

    def test_identical_boxes_map_to_themselves(self):
        from tools.planner.common.metrics import match_slots

        boxes = np.array([[0., 0., 10., 10.], [50., 50., 60., 60.]])
        np.testing.assert_array_equal(match_slots(boxes, boxes), [0, 1])

    def test_swapped_slots_are_matched_back(self):
        from tools.planner.common.metrics import match_slots

        gt = np.array([[0., 0., 10., 10.], [50., 50., 60., 60.]])
        np.testing.assert_array_equal(match_slots(gt[::-1], gt), [1, 0])

    def test_error_is_zero_when_the_prediction_is_exact(self):
        from tools.planner.common.metrics import bbox_mse

        boxes = np.random.rand(4, 2, 4) * 100
        self.assertAlmostEqual(bbox_mse(boxes, boxes)["mean_mse"], 0.0)

    def test_matching_absorbs_a_slot_permutation(self):
        from tools.planner.common.metrics import bbox_mse

        gt = np.random.rand(3, 2, 4) * 100
        swapped = gt[:, ::-1, :]
        self.assertAlmostEqual(bbox_mse(swapped, gt)["mean_mse"], 0.0)

    def test_mismatched_shapes_are_rejected(self):
        from tools.planner.common.metrics import bbox_mse

        with self.assertRaises(ValueError):
            bbox_mse(np.zeros((3, 2, 4)), np.zeros((4, 2, 4)))

    def test_temporal_order_is_one_when_the_plan_runs_forwards(self):
        from tools.planner.common.metrics import temporal_order

        gt = np.zeros((4, 1, 4))
        for t in range(4):
            gt[t, 0] = [t * 10, 0, t * 10 + 5, 5]

        self.assertAlmostEqual(temporal_order(gt, gt), 1.0)

    def test_temporal_order_is_negative_when_the_plan_runs_backwards(self):
        from tools.planner.common.metrics import temporal_order

        gt = np.zeros((4, 1, 4))
        for t in range(4):
            gt[t, 0] = [t * 10, 0, t * 10 + 5, 5]

        self.assertLess(temporal_order(gt[::-1], gt), 0)

    def test_score_flags_a_prediction_that_beats_the_straight_line(self):
        from tools.planner.common.metrics import score_window

        gt = np.array([[[10., 10., 20., 20.]], [[20., 20., 30., 30.]]])
        good = gt.copy()
        bad = gt + 50.0

        scores = score_window(good, gt, baseline_trace=bad)
        self.assertTrue(scores["beats_baseline"])
        self.assertLess(scores["mse_ratio"], 1.0)

    def test_summary_records_whether_the_plan_was_the_expected_length(self):
        from tools.planner.common.metrics import summarize

        window = {"init": 0, "goal": 4, "intermediate": [1, 2, 3],
                  "plan_length": 4}

        self.assertTrue(summarize(True, 4, 1.0, window)["plan_length_match"])
        self.assertFalse(summarize(True, 2, 1.0, window)["plan_length_match"])


if __name__ == "__main__":
    unittest.main()
