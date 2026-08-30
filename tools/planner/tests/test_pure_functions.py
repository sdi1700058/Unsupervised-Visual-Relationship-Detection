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


def _has_pillow():
    """`puzzle_labeled_objects` imports PIL, so the scaler test needs it."""
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False

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

    def test_predicate_tier_separates_coupled_from_configurational(self):
        """Criterion 0: does one object's motion depend on the other's?

        A coupled predicate defines A's motion relative to B's *changing*
        position, so it implies a rule. A configurational one only says where
        the two sit. FOSAE needs the first kind; VidVRD is mostly the second.
        """
        from tools.video.screen_vidvrd import predicate_tier

        for p in ("chase", "follow", "move_toward", "walk_away", "run_past",
                  "move_with", "bite", "ride", "fight", "toward", "away"):
            self.assertEqual(predicate_tier(p), "coupled", p)

        for p in ("stand_left", "walk_behind", "sit_above", "next_to",
                  "lie_beneath", "stand_inside", "front", "right"):
            self.assertEqual(predicate_tier(p), "configurational", p)

        for p in ("larger", "taller", "faster"):
            self.assertEqual(predicate_tier(p), "attribute", p)

    def test_predicate_tier_prefers_the_coupled_reading(self):
        """`walk_past` is motion relative to a moving reference, not a place.

        The suffix decides, and a coupled suffix wins over the verb. This is
        the judgement most likely to be wrong, so it is pinned by a test where
        it can be found and changed.
        """
        from tools.video.screen_vidvrd import predicate_tier

        self.assertEqual(predicate_tier("walk_past"), "coupled")
        self.assertEqual(predicate_tier("stand_next_to"), "configurational")

    def test_every_predicate_classifies(self):
        from tools.video.screen_vidvrd import predicate_tier, PREDICATE_TIERS

        # No predicate may fall through to "other": an unclassified predicate
        # is silently excluded from the structure score.
        for group in PREDICATE_TIERS.values():
            for p in group:
                self.assertIn(predicate_tier(p),
                              ("coupled", "configurational", "attribute"))

    def test_coupled_coverage_is_a_fraction_of_the_clip(self):
        from tools.video.screen_vidvrd import coupled_coverage

        # A ten-frame clip with one coupled relation over frames 2..6.
        doc = {"trajectories": [[]] * 10,
               "relation_instances": [
                   {"predicate": "chase", "begin_fid": 2, "end_fid": 7},
                   {"predicate": "stand_left", "begin_fid": 0, "end_fid": 10}]}
        self.assertAlmostEqual(coupled_coverage(doc), 0.5)

    def test_coupled_coverage_is_zero_without_coupled_relations(self):
        from tools.video.screen_vidvrd import coupled_coverage

        doc = {"trajectories": [[]] * 8,
               "relation_instances": [
                   {"predicate": "taller", "begin_fid": 0, "end_fid": 8}]}
        self.assertEqual(coupled_coverage(doc), 0.0)

    @unittest.skipUnless(_has_pillow(), "pillow is not installed in this "
                         "interpreter; run under .venv-local")
    def test_canvas_scaler_loads_without_the_training_stack(self):
        """The oracle claims to need no keras, and it should be true.

        `puzzle_labeled_objects` imports only os, json, numpy and PIL, but
        importing it through the package runs `latplan/__init__.py`, which
        pulls in TensorFlow. That blocked every local oracle run. The loader
        reaches the module directly, so the same function is imported rather
        than copied.
        """
        from tools.planner.oracle import load_canvas_scaler

        scale, width, height = load_canvas_scaler()
        self.assertEqual((height, width), (200, 300))
        # A box filling the left half of a 1280x576 frame maps to the left
        # half of the canvas.
        self.assertEqual(scale([0, 0, 640, 288], 1280, 576), (0, 0, 150, 100))

    def test_report_verdict_reads_a_win(self):
        """The report must state the bottom line before any table.

        Its whole purpose is that a result can be understood without reading
        a CSV, so the verdict is computed rather than left to the reader.
        """
        from tools.planner.make_report import summarise

        rows = [
            {"reachability": "True", "moving_gt_steps": "7", "bbox_mse": "12.0",
             "baseline_mse": "327.0", "mse_ratio": "0.046",
             "beats_baseline": "True", "bbox_iou": "0.93",
             "baseline_iou": "0.86", "plan_length": "7",
             "decode_fallbacks": "0", "init": "0", "goal": "7"},
            {"reachability": "True", "moving_gt_steps": "7", "bbox_mse": "11.0",
             "baseline_mse": "944.0", "mse_ratio": "0.012",
             "beats_baseline": "True", "bbox_iou": "0.94",
             "baseline_iou": "0.85", "plan_length": "7",
             "decode_fallbacks": "0", "init": "7", "goal": "14"},
        ]
        s = summarise(rows)
        self.assertEqual(s["solved"], 2)
        self.assertEqual(s["scored"], 2)
        self.assertAlmostEqual(s["ratio"], 0.029, places=3)
        self.assertEqual(s["beats"], 2)
        self.assertIn("beats", s["verdict"].lower())

    def test_report_verdict_reads_a_loss(self):
        from tools.planner.make_report import summarise

        rows = [{"reachability": "True", "moving_gt_steps": "7",
                 "bbox_mse": "500.0", "baseline_mse": "10.0",
                 "mse_ratio": "50.0", "beats_baseline": "False",
                 "bbox_iou": "0.1", "baseline_iou": "0.8",
                 "plan_length": "3", "decode_fallbacks": "2",
                 "init": "0", "goal": "7"}]
        s = summarise(rows)
        self.assertEqual(s["beats"], 0)
        self.assertNotIn("beats the", s["verdict"].lower())

    def test_report_handles_a_run_where_nothing_solved(self):
        from tools.planner.make_report import summarise

        rows = [{"reachability": "False", "moving_gt_steps": "7",
                 "bbox_mse": "", "baseline_mse": "", "mse_ratio": "",
                 "beats_baseline": "", "bbox_iou": "", "baseline_iou": "",
                 "plan_length": "0", "decode_fallbacks": "0",
                 "init": "0", "goal": "7"}]
        s = summarise(rows)
        self.assertEqual(s["solved"], 0)
        self.assertIsNone(s["ratio"])
        self.assertIn("no window", s["verdict"].lower())

    def test_report_excludes_windows_without_motion(self):
        """A window whose boxes never move scores nothing worth reporting."""
        from tools.planner.make_report import summarise

        rows = [{"reachability": "True", "moving_gt_steps": "0",
                 "bbox_mse": "1.0", "baseline_mse": "0.0001",
                 "mse_ratio": "10000", "beats_baseline": "False",
                 "bbox_iou": "0.9", "baseline_iou": "0.9",
                 "plan_length": "1", "decode_fallbacks": "0",
                 "init": "0", "goal": "7"}]
        s = summarise(rows)
        self.assertEqual(s["solved"], 1)
        self.assertEqual(s["scored"], 0)

    @unittest.skipUnless(_has_pillow(), "pillow is not installed in this "
                         "interpreter; run under .venv-local")
    def test_something_else_frames_keep_slot_identity(self):
        """`standard_category` is the slot id, and it must drive the columns.

        Something-Else labels each box with `standard_category` — `0000`,
        `0001`, `hand`. Ordering columns by anything else (list position, say)
        would let a slot swap between frames, which the Hungarian matching
        would then have to undo, and it would silently corrupt every
        trajectory.
        """
        from tools.planner.oracle import boxes_from_something_else_frames

        def box(x, cat):
            return {"box2d": {"x1": x, "y1": 0.0, "x2": x + 10.0, "y2": 10.0},
                    "standard_category": cat}

        # The hand is listed first in one frame and second in the next.
        frames = [
            {"labels": [box(0, "0000"), box(50, "hand")], "nr_instances": 2},
            {"labels": [box(60, "hand"), box(10, "0000")], "nr_instances": 2},
        ]
        boxes, meta = boxes_from_something_else_frames(
            frames, width=100, height=100, num_objs=2)

        self.assertEqual(boxes.shape, (2, 2, 4))
        self.assertEqual(meta["slots"], ["0000", "hand"])
        # Slot 0 is the object, which moved 0 -> 10, not the hand.
        self.assertLess(boxes[0][0][0], boxes[1][0][0])
        self.assertGreater(boxes[0][1][0], boxes[0][0][0])

    @unittest.skipUnless(_has_pillow(), "pillow is not installed in this "
                         "interpreter; run under .venv-local")
    def test_something_else_absent_slot_is_zero(self):
        """A slot missing from a frame stays all-zero, meaning 'not here'."""
        from tools.planner.oracle import boxes_from_something_else_frames

        frames = [
            {"labels": [{"box2d": {"x1": 1., "y1": 1., "x2": 9., "y2": 9.},
                         "standard_category": "0000"},
                        {"box2d": {"x1": 20., "y1": 1., "x2": 30., "y2": 9.},
                         "standard_category": "hand"}]},
            {"labels": [{"box2d": {"x1": 2., "y1": 1., "x2": 9., "y2": 9.},
                         "standard_category": "0000"}]},
        ]
        boxes, meta = boxes_from_something_else_frames(
            frames, width=100, height=100, num_objs=2)
        hand = meta["slots"].index("hand")
        self.assertEqual(list(boxes[1][hand]), [0, 0, 0, 0])
        self.assertEqual(meta["absent"], 1)

    def test_action_effect_consistency_is_high_when_effects_repeat(self):
        """A STRIPS operator has one effect. This measures whether one exists.

        `RELATED_WORK.md` A4, A6 and A7 all argue a representation is
        plannable when the same action flips the same bits wherever it
        applies. Nothing in the pipeline measured that, so a model could score
        well on reconstruction while its "actions" were all idiosyncratic.
        """
        from tools.planner.common.metrics import action_effect_consistency

        # Two actions. Each flips its own pair of bits, every time.
        z = np.array([
            [0, 0, 0, 0], [1, 1, 0, 0],   # action A
            [0, 0, 1, 1], [1, 1, 1, 1],   # action A again, same effect
            [0, 0, 0, 0], [0, 0, 1, 1],   # action B
            [1, 1, 0, 0], [1, 1, 1, 1],   # action B again
        ], dtype=np.int8)
        pre, suc = z[0::2], z[1::2]
        labels = ["A", "A", "B", "B"]

        r = action_effect_consistency(pre, suc, labels)
        self.assertAlmostEqual(r["within"], 1.0, places=6)
        self.assertAlmostEqual(r["between"], 0.0, places=6)
        self.assertAlmostEqual(r["consistency"], 1.0, places=6)

    def test_action_effect_consistency_is_low_when_effects_are_arbitrary(self):
        from tools.planner.common.metrics import action_effect_consistency

        # The same label, but every transition flips different bits.
        z = np.array([
            [0, 0, 0, 0], [1, 0, 0, 0],
            [0, 0, 0, 0], [0, 1, 0, 0],
            [0, 0, 0, 0], [0, 0, 1, 0],
            [0, 0, 0, 0], [0, 0, 0, 1],
        ], dtype=np.int8)
        # Two labels, not one. With a single label there is no `between` set,
        # so `within - between` reduces to `within` and the measure cannot
        # fail; `consistency` is None in that case since 2026-08-30.
        r = action_effect_consistency(z[0::2], z[1::2], ["A", "A", "B", "B"])
        self.assertAlmostEqual(r["within"], 0.0, places=6)
        self.assertLess(r["consistency"], 0.01)

    def test_action_effect_consistency_needs_a_repeated_label(self):
        from tools.planner.common.metrics import action_effect_consistency

        z = np.array([[0, 0], [1, 0], [0, 0], [0, 1]], dtype=np.int8)
        r = action_effect_consistency(z[0::2], z[1::2], ["A", "B"])
        # One transition per label: nothing to compare within an action.
        self.assertIsNone(r["consistency"])

    def test_binary_encoding_round_trips_like_one_hot(self):
        """A second positional code, chosen because one-hot is the worst.

        `EVAL.md` measures one-hot at action-effect consistency 0.000 — every
        start position gives "move one bin" a different bit-effect — against
        0.298 for a binary code. This adds the binary option so the claim can
        be tested end to end. Both codes must decode to the same boxes.
        """
        from tools.planner.oracle import boxes_to_latents, latents_to_boxes

        boxes = np.array([[[10., 20., 60., 80.], [100., 30., 150., 90.]]])
        for encoding in ("onehot", "binary"):
            z = boxes_to_latents(boxes, 60, 40, encoding=encoding)
            back = latents_to_boxes(z, 2, 60, 40, encoding=encoding)
            self.assertEqual(back.shape, boxes.shape, encoding)
            # Within one bin: 300/60 = 5 px in x, 200/40 = 5 px in y.
            self.assertTrue(np.all(np.abs(back - boxes) <= 6), encoding)

    def test_binary_encoding_is_far_smaller(self):
        from tools.planner.oracle import boxes_to_latents

        boxes = np.zeros((1, 2, 4))
        boxes[0, 0] = [10., 20., 60., 80.]
        wide = boxes_to_latents(boxes, 60, 40, encoding="onehot")
        tight = boxes_to_latents(boxes, 60, 40, encoding="binary")
        # 200 bits per object against 24: fewer propositions for the planner.
        self.assertEqual(wide.shape[1], 2 * 200)
        self.assertLess(tight.shape[1], wide.shape[1] // 4)

    def test_binary_encoding_keeps_an_absent_slot_empty(self):
        """A padded slot must stay all-zero, or the oracle invents an object."""
        from tools.planner.oracle import boxes_to_latents

        boxes = np.array([[[10., 20., 60., 80.], [0., 0., 0., 0.]]])
        z = boxes_to_latents(boxes, 60, 40, encoding="binary")
        per = z.shape[1] // 2
        self.assertEqual(int(z[0, per:].sum()), 0)
        self.assertGreater(int(z[0, :per].sum()), 0)

    def test_binary_encoding_gives_an_action_one_effect_more_often(self):
        """The point of the option, stated as a test."""
        from tools.planner.oracle import boxes_to_latents
        from tools.planner.common.metrics import action_effect_consistency

        # One object sliding right one bin at a time, from many start points.
        pre, suc, labels = [], [], []
        for start in range(0, 40):
            a = np.array([[[start * 5., 50., start * 5. + 20., 90.]]])
            b = np.array([[[(start + 1) * 5., 50., (start + 1) * 5. + 20., 90.]]])
            pre.append(a)
            suc.append(b)
            labels.append("right")

        scores = {}
        for encoding in ("onehot", "binary"):
            p = np.concatenate([boxes_to_latents(x, 60, 40, encoding=encoding)
                                for x in pre])
            s = np.concatenate([boxes_to_latents(x, 60, 40, encoding=encoding)
                                for x in suc])
            scores[encoding] = action_effect_consistency(p, s, labels)["within"]
        self.assertGreater(scores["binary"], scores["onehot"])

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
        # None, not 0.0. Zero is the BEST possible score, so returning it here
        # put a window with no data at all into the reported median as a
        # perfect result.
        self.assertIsNone(result["mean_mse"])

    def test_score_window_drops_objects_absent_at_an_endpoint(self):
        """The baseline cannot be formed from an endpoint that does not exist.

        `linear_interp_bboxes` interpolates raw boxes. If an object is absent
        at a window endpoint its box is zeros, so the straight line starts at
        the origin and the baseline error is enormous — which **flatters the
        planner**. Those object-frames are unscoreable for both sides, not just
        for one, so the exclusion has to be shared.
        """
        from tools.planner.common.metrics import score_window

        # Two objects. Object 1 is absent at the init endpoint.
        init = np.array([[0., 0., 10., 10.], [0., 0., 0., 0.]])
        goal = np.array([[20., 0., 30., 10.], [50., 50., 60., 60.]])
        gt = np.array([
            [[10., 0., 20., 10.], [40., 40., 50., 50.]],
        ])
        pred = np.array([
            [[10., 0., 20., 10.], [99., 99., 109., 109.]],
        ])
        baseline = np.array([
            [[10., 0., 20., 10.], [25., 25., 30., 30.]],
        ])

        scored = score_window(pred, gt, baseline, matching="fixed",
                              endpoints=(init, goal))
        # Only object 0 is scoreable, and both sides predict it exactly.
        self.assertEqual(scored["planner"]["mean_mse"], 0.0)
        self.assertEqual(scored["baseline_linear"]["mean_mse"], 0.0)

    def test_score_window_without_endpoints_scores_everything(self):
        from tools.planner.common.metrics import score_window

        gt = np.array([[[10., 0., 20., 10.]]])
        pred = np.array([[[10., 0., 20., 10.]]])
        base = np.array([[[12., 0., 22., 10.]]])
        scored = score_window(pred, gt, base, matching="fixed")
        self.assertEqual(scored["planner"]["mean_mse"], 0.0)
        self.assertGreater(scored["baseline_linear"]["mean_mse"], 0.0)

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


class TestReviewRegressions(unittest.TestCase):
    """Defects found by the 2026-08-30 review, each confirmed by execution.

    Every one of these is a *silent wrong number* rather than a crash, which
    is the failure mode this project keeps hitting. They are grouped so the
    next reviewer can see at a glance what has already been paid for.
    """

    # ── metrics: an unscoreable window must not score as perfect ──────────
    def test_window_with_no_scoreable_object_reports_none_not_zero(self):
        from tools.planner.common.metrics import bbox_mse

        pred = np.full((4, 2, 4), 400.0)     # wildly wrong everywhere
        gt = np.zeros((4, 2, 4))             # every object absent

        out = bbox_mse(pred, gt, matching="fixed")

        # 0.0 is the BEST possible score, so returning it for "no data at
        # all" put a perfect window into the reported median.
        self.assertIsNone(out["mean_mse"])
        self.assertEqual(out["skipped_absent"], 8)

    # ── metrics: the pairing must not be solved against absent boxes ──────
    def test_match_slots_ignores_absent_ground_truth(self):
        from tools.planner.common.metrics import match_slots

        # gt object 0 is present and matches pred slot 0 exactly.
        # gt object 1 is ABSENT, and its all-zero box is a strong attractor
        # for pred slot 1, which is also empty.
        pred = np.array([[20.0, 20, 30, 30], [0, 0, 0, 0]])
        gt = np.array([[20.0, 20, 30, 30], [0, 0, 0, 0]])

        mapping = match_slots(pred, gt, gt_present=np.array([True, False]))
        self.assertEqual(int(mapping[0]), 0)

    def test_match_slots_pairs_on_a_frame_where_the_object_is_present(self):
        from tools.planner.common.metrics import match_slots

        # Frame 0 has nothing annotated; the pairing must come from later
        # frames rather than from a frame of zeros.
        pred = np.zeros((3, 2, 4))
        gt = np.zeros((3, 2, 4))
        pred[1:, 0] = [10.0, 10, 20, 20]
        pred[1:, 1] = [90.0, 90, 99, 99]
        gt[1:, 0] = [90.0, 90, 99, 99]
        gt[1:, 1] = [10.0, 10, 20, 20]

        mapping = match_slots(pred, gt)
        self.assertEqual([int(v) for v in mapping], [1, 0])

    def test_bbox_mse_does_not_reward_a_wrong_pairing(self):
        from tools.planner.common.metrics import bbox_mse

        pred = np.array([[[20.0, 20, 30, 30], [0, 0, 0, 0]]])
        gt = np.array([[[20.0, 20, 30, 30], [0, 0, 0, 0]]])

        out = bbox_mse(pred, gt, matching="hungarian")
        self.assertEqual(out["mean_mse"], 0.0)
        self.assertEqual(int(out["mapping"][0]), 0)

    # ── metrics: a near-zero baseline must not produce a huge ratio ───────
    def test_mse_ratio_is_none_when_the_baseline_is_degenerate(self):
        from tools.planner.common.metrics import mse_ratio

        self.assertIsNone(mse_ratio(12.0, 4.1e-10))
        self.assertIsNone(mse_ratio(12.0, 0.0))
        self.assertAlmostEqual(mse_ratio(12.0, 24.0), 0.5)

    # ── metrics: motion must not be manufactured by an object vanishing ───
    def test_moving_gt_steps_ignores_appearance_and_disappearance(self):
        from tools.planner.common.metrics import moving_gt_steps

        static = np.tile(np.array([[[10.0, 10, 20, 20]]]), (4, 1, 1))
        self.assertEqual(moving_gt_steps(static), 0)

        vanishes = static.copy()
        vanishes[2:] = 0.0          # object leaves; not motion
        self.assertEqual(moving_gt_steps(vanishes), 0)

    # ── metrics: a single action label cannot be "consistent" ─────────────
    def test_action_effect_consistency_is_none_for_a_single_label(self):
        from tools.planner.common.metrics import action_effect_consistency

        rng = np.random.RandomState(0)
        pre = rng.randint(0, 2, size=(8, 12)).astype(np.int8)
        suc = rng.randint(0, 2, size=(8, 12)).astype(np.int8)

        out = action_effect_consistency(pre, suc, ["a"] * 8)
        self.assertIsNone(out["consistency"])

    # ── oracle: the binary code must hold every bin it claims to ──────────
    def test_binary_code_survives_power_of_two_bin_counts(self):
        from tools.planner import oracle as O

        # A box sitting in the TOP bin on every coordinate. Before the fix
        # the code overflowed its field, the object block went all-zero, and
        # `latents_to_boxes` read the object as ABSENT — the oracle deleted
        # an object that was there.
        box = np.array([[[290.0, 190.0, 299.0, 199.0]]])
        for bins in (8, 16, 32, 64):
            lat = O.boxes_to_latents(box, bins_x=bins, bins_y=bins,
                                     encoding="binary")
            self.assertGreater(lat.sum(), 0,
                               "object deleted at bins=%d" % bins)
            back = O.latents_to_boxes(lat, 1, bins_x=bins, bins_y=bins,
                                      encoding="binary")
            # x2 >= x1, not x2 > x1: at 8 bins the two coordinates legitimately
            # quantise into the same bin. The defect was inversion — x2 came
            # back as 4.7 px against an x1 of 285.9 — not equality.
            self.assertGreaterEqual(back[0, 0, 2], back[0, 0, 0],
                                    "box inverted at bins=%d" % bins)
            self.assertGreaterEqual(back[0, 0, 3], back[0, 0, 1],
                                    "box inverted at bins=%d" % bins)

    def test_code_width_holds_the_offset_value(self):
        from tools.planner.oracle import _code_width

        for bins in (2, 8, 16, 31, 32, 33, 64):
            self.assertGreaterEqual(2 ** _code_width(bins) - 1, bins,
                                    "width too small at bins=%d" % bins)

    # ── oracle: the floor must be the floor the real decoder actually has ─
    def test_dequantise_matches_the_real_decoder(self):
        from tools.planner import oracle as O

        # `common/decode.py` maps a bin index to the bin's LEFT EDGE:
        #     x1 = argmax(...) * (canvas_w / X)
        # The oracle used bin centres, which put its floor 4x below the one
        # every trained model is measured against.
        idx = np.array([0, 1, 5, 59])
        expected = idx * (300.0 / 60)
        np.testing.assert_allclose(O.dequantise(idx, 60, 300), expected)

    def test_round_trip_error_divides_by_present_cells_only(self):
        from tools.planner import oracle as O

        one_real = np.array([[[10.0, 10.0, 40.0, 40.0]]])
        with_padding = np.zeros((1, 3, 4))
        with_padding[0, 0] = one_real[0, 0]

        # Padding slots round-trip to exactly zero error, so averaging over
        # them reported a third of the true floor.
        self.assertAlmostEqual(O.round_trip_error(one_real),
                               O.round_trip_error(with_padding), places=6)

    # ── planner: a timeout is not a claim about the representation ────────
    def test_search_reports_timeout_separately_from_exhaustion(self):
        from tools.planner.bfs.planner import search

        z = np.zeros(24, dtype=np.int8)
        goal = np.ones(24, dtype=np.int8)
        deltas = np.eye(24, dtype=np.int8)[:6]      # cannot reach the goal

        stats = {}
        found, _, _ = search(z, goal, deltas, time_budget_s=30.0,
                             max_length=3, stats=stats)
        self.assertFalse(found)
        self.assertEqual(stats["outcome"], "exhausted")

        stats = {}
        found, _, _ = search(z, goal, deltas, time_budget_s=0.0,
                             max_length=40, stats=stats)
        self.assertFalse(found)
        self.assertEqual(stats["outcome"], "timeout")

    def test_search_is_bounded_by_a_node_cap(self):
        from tools.planner.bfs.planner import search

        # Unreachable goal, wide branching, generous clock: without a node
        # cap this grows at roughly 2.4 GiB per minute. An unbounded search
        # of exactly this shape crashed the workstation on 2026-08-28.
        z = np.zeros(64, dtype=np.int8)
        goal = np.ones(64, dtype=np.int8)
        deltas = np.eye(64, dtype=np.int8)[:20]

        stats = {}
        found, _, _ = search(z, goal, deltas, time_budget_s=30.0,
                             max_length=40, max_nodes=5000, stats=stats)
        self.assertFalse(found)
        self.assertEqual(stats["outcome"], "node_cap")

    def test_identical_endpoints_are_flagged_rather_than_counted_as_solved(self):
        from tools.planner.bfs.planner import search

        z = np.zeros(8, dtype=np.int8)
        deltas = np.eye(8, dtype=np.int8)[:3]

        stats = {}
        found, trace, _ = search(z, z.copy(), deltas, max_length=4,
                                 stats=stats)
        self.assertTrue(found)
        self.assertEqual(len(trace), 1)
        self.assertEqual(stats["outcome"], "endpoints_identical")
