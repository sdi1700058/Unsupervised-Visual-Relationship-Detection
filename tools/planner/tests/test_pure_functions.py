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
        deltas = mine_deltas(z[:-1], z[1:])
        self.assertEqual(len(deltas), 1)
        np.testing.assert_array_equal(deltas[0], [1, 0])

    def test_search_walks_one_bit_at_a_time(self):
        from tools.planner.bfs.planner import mine_deltas, search

        z = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 1],
                      [0, 1, 1, 1],
                      [1, 1, 1, 1]], dtype=np.int8)
        deltas = mine_deltas(z[:-1], z[1:])

        found, trace, _ = search(z[0], z[-1], deltas, time_budget_s=5)
        self.assertTrue(found)
        self.assertEqual(len(trace) - 1, 4)
        np.testing.assert_array_equal(trace[0], z[0])
        np.testing.assert_array_equal(trace[-1], z[-1])

    def test_goal_equal_to_init_needs_no_actions(self):
        from tools.planner.bfs.planner import mine_deltas, search

        z = np.array([[0, 0], [1, 0]], dtype=np.int8)
        found, trace, _ = search(z[0], z[0], mine_deltas(z[:-1], z[1:]), time_budget_s=5)
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

    def test_repeated_frames_do_not_make_the_window_unplannable(self):
        """A window whose latents repeat still has a plan, a shorter one.

        The trained models encode several consecutive frames to one latent, so
        a k-frame window holds fewer than k-1 real transitions. Demanding
        exactly k-1 actions rejects the true plan and sends the search after a
        padded one it cannot afford. `max_length` asks for the shortest plan
        that fits inside the window instead.
        """
        from tools.planner.bfs.planner import mine_deltas, search

        # Eight frames, but the latent only moves on three of the seven steps.
        z = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 1],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1],
                      [0, 0, 1, 1],
                      [0, 1, 1, 1],
                      [0, 1, 1, 1]], dtype=np.int8)
        deltas = mine_deltas(z[:-1], z[1:])

        # The window is 8 frames, so the old code asked for exactly 7 actions.
        # Here that is satisfiable only by padding: the plan wanders away and
        # comes back, because a delta applied twice is the identity. On the
        # real exports the delta set is large enough that the same search runs
        # out of budget instead, which is the failure this fixes.
        found, padded, _ = search(z[0], z[-1], deltas, time_budget_s=5,
                                  exact_length=7)
        self.assertTrue(found)
        self.assertEqual(len(padded) - 1, 7)
        self.assertLess(len({row.tobytes() for row in padded}), len(padded),
                        "a padded plan has to revisit a state")

        # The honest question is whether the goal is reachable within 7.
        found, trace, _ = search(z[0], z[-1], deltas, time_budget_s=5,
                                 max_length=7)
        self.assertTrue(found)
        self.assertEqual(len(trace) - 1, 3)
        np.testing.assert_array_equal(trace[0], z[0])
        np.testing.assert_array_equal(trace[-1], z[-1])

    def test_max_length_refuses_a_plan_that_does_not_fit(self):
        from tools.planner.bfs.planner import search

        # Three separate bits to set, one delta each, so no plan under three.
        deltas = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)
        found, _, _ = search(np.zeros(3, dtype=np.int8),
                             np.ones(3, dtype=np.int8),
                             deltas, time_budget_s=2, max_length=2)
        self.assertFalse(found)

    def test_moving_gt_steps_counts_frames_where_the_boxes_move(self):
        """A window whose annotated boxes never move cannot measure anything.

        Clip 00005005 opens with 60 consecutive frame pairs of completely
        motionless boxes, so the first eight windows of that export score
        nothing but quantisation noise. The count has to come from the
        annotations, not from the latents: a model can change its code while
        the objects stand still.
        """
        from tools.planner.common.metrics import moving_gt_steps

        boxes = np.array([
            [[0., 0., 10., 10.]],
            [[0., 0., 10., 10.]],   # still
            [[2., 0., 12., 10.]],   # moved
            [[2., 0., 12., 10.]],   # still
            [[5., 1., 15., 11.]],   # moved
        ])
        self.assertEqual(moving_gt_steps(boxes), 2)
        self.assertEqual(moving_gt_steps(boxes[:2]), 0)
        self.assertEqual(moving_gt_steps(boxes[:1]), 0)

    def test_moving_gt_steps_ignores_movement_below_the_tolerance(self):
        from tools.planner.common.metrics import moving_gt_steps

        boxes = np.array([
            [[0., 0., 10., 10.]],
            [[0.0001, 0., 10., 10.]],
        ])
        self.assertEqual(moving_gt_steps(boxes), 0)
        self.assertEqual(moving_gt_steps(boxes, tol=0.0), 1)

    def test_latent_geometry_rewards_a_position_aligned_code(self):
        """A code whose Hamming distance tracks position scores near 1.

        This is what decides whether a plan's intermediate states decode to
        sensible boxes. Those states are almost never latents the model
        produced, so `Export.boxes_for` falls back to the Hamming-nearest
        observed latent. If Hamming distance tracks position that fallback is
        harmless; if it does not, the decoded box is arbitrary.
        """
        from tools.planner.common.metrics import latent_geometry

        # A unary code: frame i has i bits set, and its box sits at x = i.
        n = 8
        z = np.array([[1] * i + [0] * (n - i) for i in range(n)], dtype=np.int8)
        boxes = np.array([[[float(i), 0., float(i) + 1., 1.]] for i in range(n)])

        g = latent_geometry(z, boxes)
        self.assertGreater(g["spearman"], 0.9)
        # Every frame's Hamming-nearest neighbour is an adjacent frame, one
        # unit away in x, so the corner distance is sqrt(1 + 1) over two
        # corners moved by 1 each.
        self.assertLess(g["nearest_box_error"], 2.0)

    def test_latent_geometry_penalises_a_shuffled_code(self):
        from tools.planner.common.metrics import latent_geometry

        n = 8
        # Same boxes, but the codes are assigned in an order unrelated to x.
        order = [0, 5, 2, 7, 4, 1, 6, 3]
        z = np.array([[1] * order[i] + [0] * (n - order[i]) for i in range(n)],
                     dtype=np.int8)
        boxes = np.array([[[float(i), 0., float(i) + 1., 1.]] for i in range(n)])

        shuffled = latent_geometry(z, boxes)
        self.assertLess(shuffled["spearman"], 0.6)

    def test_latent_geometry_handles_a_degenerate_code(self):
        from tools.planner.common.metrics import latent_geometry

        # Every frame encodes identically. No pair carries information, so the
        # correlation is undefined rather than zero.
        z = np.zeros((5, 4), dtype=np.int8)
        boxes = np.array([[[float(i), 0., 1., 1.]] for i in range(5)])
        g = latent_geometry(z, boxes)
        self.assertIsNone(g["spearman"])

    def test_clip_stats_counts_filled_frames_and_real_motion(self):
        """Frames with no annotation are the ones `--fill-annotations` invents.

        VidVRD annotates a subset of frames. Carrying the last box forward
        into the rest makes those transitions identical on both sides, so they
        teach "nothing changed". The statistic has to separate annotated
        frames from filled ones, and measure motion only across annotated
        pairs.
        """
        from tools.video.screen_vidvrd import clip_stats

        # Six frames; frames 2 and 3 carry no annotation at all.
        def box(x):
            return {"tid": 0, "bbox": {"xmin": x, "ymin": 0,
                                       "xmax": x + 10, "ymax": 10}}
        trajectories = [[box(0)], [box(5)], [], [], [box(40)], [box(50)]]

        s = clip_stats({"video_id": "v", "trajectories": trajectories,
                        "width": 100, "height": 100})
        self.assertEqual(s["frames"], 6)
        self.assertEqual(s["annotated"], 4)
        self.assertAlmostEqual(s["fill_frac"], 2 / 6.0)
        self.assertEqual(s["n_objects"], 1)
        # Consecutive annotated pairs are 0->1 and 4->5. Each corner moves by
        # 5 and 10, so the summed absolute displacement is 10 and 20.
        self.assertAlmostEqual(s["mean_disp"], 15.0)

    def test_clip_stats_on_a_fully_annotated_clip_reports_no_fill(self):
        from tools.video.screen_vidvrd import clip_stats

        def box(x):
            return {"tid": 0, "bbox": {"xmin": x, "ymin": 0,
                                       "xmax": x + 1, "ymax": 1}}
        s = clip_stats({"video_id": "v",
                        "trajectories": [[box(i)] for i in range(5)],
                        "width": 10, "height": 10})
        self.assertEqual(s["fill_frac"], 0.0)
        self.assertEqual(s["annotated"], 5)

    def test_clip_stats_survives_a_clip_with_no_boxes(self):
        from tools.video.screen_vidvrd import clip_stats

        s = clip_stats({"video_id": "v", "trajectories": [[], []],
                        "width": 10, "height": 10})
        self.assertIsNone(s)

    def test_nonlinearity_is_zero_for_constant_velocity(self):
        """Straight-line motion is exactly what interpolation predicts.

        `EVAL.md` §4.9: a planner can only beat linear interpolation where the
        motion is not linear. This scores how much of a clip's motion the
        straight line already explains, so clips can be screened for the
        property before any are baked.
        """
        from tools.video.screen_vidvrd import window_nonlinearity

        # One object, constant velocity: interpolation is exact.
        track = {i: (float(i), 0.0, float(i) + 10.0, 10.0) for i in range(9)}
        self.assertAlmostEqual(window_nonlinearity({0: track}, window=5), 0.0)

    def test_nonlinearity_is_positive_for_a_step_change(self):
        from tools.video.screen_vidvrd import window_nonlinearity

        # Still, then one jump, then still. Interpolation smears the jump.
        xs = [0., 0., 0., 0., 40., 40., 40., 40., 40.]
        track = {i: (x, 0.0, x + 10.0, 10.0) for i, x in enumerate(xs)}
        self.assertGreater(window_nonlinearity({0: track}, window=5), 0.05)

    def test_nonlinearity_ignores_windows_that_are_not_contiguous(self):
        from tools.video.screen_vidvrd import window_nonlinearity

        # Frames 3 and 4 are missing, so no window of 5 is contiguous.
        track = {i: (float(i), 0.0, 1.0, 1.0) for i in (0, 1, 2, 5, 6, 7)}
        self.assertIsNone(window_nonlinearity({0: track}, window=5))

    def test_crossover_ratio_is_below_one_when_motion_is_large(self):
        """`EVAL.md` §4.2's criterion, computed per clip from annotation alone.

        `mse_ratio < 1` is arithmetically unreachable while the quantisation
        floor exceeds the linear-interpolation baseline. Both sides are
        computable without a model, so a clip can be judged before it is baked.
        """
        from tools.video.screen_vidvrd import window_crossover

        # A big detour: the object goes far off the straight line between the
        # endpoints, so interpolation is badly wrong and the floor is small
        # relative to it.
        xs = [0., 0., 0., 0., 200., 0., 0., 0., 0.]
        track = {i: (x, 0.0, x + 20.0, 20.0) for i, x in enumerate(xs)}
        ratio = window_crossover({0: track}, width=640, height=480, window=8)
        self.assertIsNotNone(ratio)
        self.assertLess(ratio, 1.0)

    def test_crossover_ratio_is_large_when_motion_is_a_straight_line(self):
        from tools.video.screen_vidvrd import window_crossover

        # Constant velocity: interpolation is exact, so the baseline error is
        # ~0 and the floor dominates however small it is.
        track = {i: (float(i), 0.0, float(i) + 20.0, 20.0) for i in range(9)}
        ratio = window_crossover({0: track}, width=640, height=480, window=8)
        self.assertGreater(ratio, 1.0)

    def test_crossover_ignores_windows_where_an_object_comes_and_goes(self):
        """An object that disappears must not be scored as a huge movement.

        The loader pads a missing object with a zero box, so a naive
        computation sees the box jump from its real position to the origin.
        That inflates the linear-interpolation baseline and makes a slow clip
        look winnable. Measured on real data: the three top-ranked clips all
        had an object present in only 50-67% of frames.
        """
        from tools.video.screen_vidvrd import window_crossover

        # tid 0 is present throughout and barely moves, so a straight line
        # predicts it almost exactly. tid 1 vanishes half way.
        steady = {i: (100.0 + i, 100.0, 120.0 + i, 120.0) for i in range(9)}
        vanishing = {i: (300.0, 300.0, 320.0, 320.0) for i in range(5)}

        both = window_crossover({0: steady, 1: vanishing},
                                width=640, height=480, window=8)
        alone = window_crossover({0: steady}, width=640, height=480, window=8)

        # Scoring only the object that is present throughout must give the
        # same answer as if the vanishing one had never been in the file.
        self.assertIsNotNone(both)
        self.assertAlmostEqual(both, alone, places=6)

    def test_crossover_returns_none_without_a_contiguous_window(self):
        from tools.video.screen_vidvrd import window_crossover

        track = {i: (float(i), 0.0, 1.0, 1.0) for i in (0, 1, 2, 9, 10)}
        self.assertIsNone(window_crossover({0: track}, width=64,
                                           height=48, window=8))

    def test_repeated_searches_return_the_same_plan(self):
        from tools.planner.bfs.planner import mine_deltas, search

        z = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                     dtype=np.int8)
        deltas = mine_deltas(z[:-1], z[1:])

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

    def test_mse_does_not_score_a_frame_where_the_object_is_absent(self):
        """An absent object is a zero box, and that is not a position.

        `bbox_iou` already refuses to score a padded slot; `bbox_mse` used to
        square the difference against the origin and call it error. On 23% of
        the winnable VidVRD clips at least one object does not span every
        frame, so this is not a corner case.
        """
        from tools.planner.common.metrics import bbox_mse

        gt = np.array([
            [[10., 10., 20., 20.]],
            [[0., 0., 0., 0.]],        # object absent in this frame
            [[30., 30., 40., 40.]],
        ])
        pred = np.array([
            [[10., 10., 20., 20.]],    # exact
            [[99., 99., 99., 99.]],    # scored against nothing
            [[30., 30., 40., 40.]],    # exact
        ])

        result = bbox_mse(pred, gt, matching="fixed")
        self.assertEqual(result["mean_mse"], 0.0)
        self.assertEqual(result["skipped_absent"], 1)

    def test_mse_reports_when_every_frame_is_absent(self):
        from tools.planner.common.metrics import bbox_mse

        gt = np.zeros((3, 1, 4))
        pred = np.ones((3, 1, 4)) * 5.0
        result = bbox_mse(pred, gt, matching="fixed")
        self.assertEqual(result["skipped_absent"], 3)
        self.assertEqual(result["mean_mse"], 0.0)

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


def _fake_export(tmp_dir, n_frames=12, n_bits=16, n_obj=2, linear=False):
    """Write a synthetic export so tests can run the whole chain.

    With linear=False the objects accelerate, so the straight-line baseline
    is beatable. With linear=True they move at constant speed and the
    baseline is exact, which is the case that carries no signal.
    """
    rng = np.random.default_rng(0)

    latents = np.zeros((n_frames, n_bits), dtype=np.int8)
    for t in range(1, n_frames):
        latents[t] = latents[t - 1]
        latents[t][t % n_bits] ^= 1

    gt = np.zeros((n_frames, n_obj, 4), dtype=np.float32)
    for t in range(n_frames):
        x = 10 + (2.0 * t if linear else 2.0 * t * t)
        y = 200 - (1.5 * t if linear else 1.5 * t * t)
        gt[t, 0] = [x, 50, x + 30, 90]
        gt[t, 1] = [y, 150, y + 40, 190]

    path = Path(tmp_dir) / "export.npz"
    np.savez_compressed(
        path,
        latents=latents,
        gt_boxes=gt,
        decoded_boxes=gt + rng.normal(0, 2, gt.shape).astype(np.float32),
        U=4, A=2, P=4, n_bits=n_bits, model_name="test",
        frame_ids=np.asarray([f"f{t:04d}" for t in range(n_frames)],
                             dtype="U256"),
        actions=np.concatenate([latents[:-1], latents[1:]], axis=1))
    return path


class TestExport(unittest.TestCase):

    def test_export_round_trips_without_pickle(self):
        from tools.planner.common.export import load

        with tempfile.TemporaryDirectory() as tmp:
            export = load(_fake_export(tmp))

        self.assertEqual(len(export), 12)
        self.assertEqual(export.n_bits, 16)
        self.assertEqual(export.parameters["U"], 4)

    def test_known_latent_returns_its_own_boxes(self):
        from tools.planner.common.export import load

        with tempfile.TemporaryDirectory() as tmp:
            export = load(_fake_export(tmp))

            boxes = export.boxes_for(export.latents[3])
            np.testing.assert_array_equal(boxes, export.decoded_boxes[3])
            self.assertEqual(export.fallback_count, 0)

    def test_unknown_latent_falls_back_and_is_counted(self):
        from tools.planner.common.export import load

        with tempfile.TemporaryDirectory() as tmp:
            export = load(_fake_export(tmp))

            stranger = np.ones(export.n_bits, dtype=np.int8)
            export.boxes_for(stranger)
            self.assertEqual(export.fallback_count, 1)

    def test_transitions_prefer_the_stored_actions(self):
        from tools.planner.common.export import load

        with tempfile.TemporaryDirectory() as tmp:
            export = load(_fake_export(tmp))

        pre, suc = export.transitions()
        self.assertEqual(pre.shape, (11, 16))
        self.assertEqual(suc.shape, (11, 16))


class TestFullChain(unittest.TestCase):
    """Run a whole window through the harness, end to end, with no keras."""

    def test_bfs_scores_a_window(self):
        from tools.planner.bfs.planner import run

        with tempfile.TemporaryDirectory() as tmp:
            export = _fake_export(tmp)
            out = Path(tmp) / "out"

            metrics = run(export_path=export, init_idx=0, goal_idx=4,
                          out_dir=out, time_budget_s=10)

            # Check the files before the temp directory goes away.
            self.assertTrue((out / "metrics.json").exists())
            self.assertTrue((out / "plan_trace.json").exists())

        self.assertTrue(metrics["reachability"])
        self.assertEqual(metrics["plan_length"], 4)
        self.assertEqual(metrics["expected_plan_length"], 4)
        self.assertTrue(metrics["plan_length_match"])
        self.assertEqual(metrics["n_intermediate"], 3)
        self.assertIsNotNone(metrics["bbox_mse_mean"])
        self.assertIsNotNone(metrics["baseline_mse_mean"])

    def test_linear_motion_leaves_the_ratio_undefined(self):
        from tools.planner.bfs.planner import run

        # A perfect straight line gives a zero-error baseline, so the ratio
        # cannot be formed. The run must still finish and say so.
        with tempfile.TemporaryDirectory() as tmp:
            export = _fake_export(tmp, linear=True)
            metrics = run(export_path=export, init_idx=0, goal_idx=4,
                          out_dir=Path(tmp) / "out", time_budget_s=10)

        self.assertTrue(metrics["reachability"])
        self.assertAlmostEqual(metrics["baseline_mse_mean"], 0.0, places=3)
        self.assertIsNone(metrics["mse_ratio"])

    def test_goal_before_init_is_rejected(self):
        from tools.planner.bfs.planner import run

        with tempfile.TemporaryDirectory() as tmp:
            export = _fake_export(tmp)
            with self.assertRaises(ValueError):
                run(export_path=export, init_idx=6, goal_idx=2,
                    out_dir=Path(tmp) / "out", time_budget_s=5)

    def test_window_outside_the_clip_is_rejected(self):
        from tools.planner.bfs.planner import run

        with tempfile.TemporaryDirectory() as tmp:
            export = _fake_export(tmp)
            with self.assertRaises(IndexError):
                run(export_path=export, init_idx=0, goal_idx=99,
                    out_dir=Path(tmp) / "out", time_budget_s=5)


class TestExportDedupe(unittest.TestCase):

    def test_repeated_effects_collapse_to_one_row(self):
        from tools.planner.export_latents import _dedupe_transitions

        base = np.array([
            [0, 1, 0, 0,  1, 1, 0, 0],   # add bit 0
            [1, 1, 0, 0,  1, 0, 0, 0],   # delete bit 1
            [0, 1, 0, 0,  0, 1, 1, 0],   # add bit 2
        ], dtype=np.int8)

        rows = np.repeat(base, 500, axis=0)
        kept = _dedupe_transitions(rows)

        self.assertEqual(len(kept), 3)
        np.testing.assert_array_equal(kept, base)

    def test_transitions_that_change_nothing_are_dropped(self):
        from tools.planner.export_latents import _dedupe_transitions

        rows = np.array([
            [1, 0,  1, 0],   # no-op
            [0, 0,  1, 0],   # add bit 0
        ], dtype=np.int8)

        kept = _dedupe_transitions(rows)
        self.assertEqual(len(kept), 1)
        np.testing.assert_array_equal(kept[0], [0, 0, 1, 0])
