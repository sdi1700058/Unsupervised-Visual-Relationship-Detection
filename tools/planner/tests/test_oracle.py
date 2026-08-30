#!/usr/bin/env python3
"""Tests for the oracle export. numpy only — no keras, no data on disk."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from tools.planner.oracle import (          # noqa: E402
    CANVAS_H, CANVAS_W, bits_per_object, boxes_to_latents, build_export,
    dequantise, latents_to_boxes, quantise, round_trip_error)


class TestQuantise(unittest.TestCase):

    def test_spans_the_full_range(self):
        self.assertEqual(quantise(0.0, 10, 100), 0)
        self.assertEqual(quantise(99.9, 10, 100), 9)

    def test_top_edge_stays_in_range(self):
        # A coordinate sitting exactly on the canvas edge must not index
        # past the last bin.
        self.assertEqual(quantise(100.0, 10, 100), 9)
        self.assertEqual(quantise(1e9, 10, 100), 9)

    def test_negative_clips_to_zero(self):
        self.assertEqual(quantise(-5.0, 10, 100), 0)

    def test_dequantise_returns_the_bin_left_edge(self):
        """Matches `common/decode.py`, which is the point of the oracle.

        This asserted the bin CENTRE until 2026-08-30. The centre is the
        better estimator, but no trained model's decoder emits it --
        `decode.py:79` takes `argmax * (canvas_w / X)`, the left edge -- so
        the oracle's floor sat 4x below the floor it was the ceiling for.
        """
        self.assertAlmostEqual(dequantise(0, 10, 100), 0.0)
        self.assertAlmostEqual(dequantise(9, 10, 100), 90.0)

    def test_round_trip_within_one_bin(self):
        """One bin, not half a bin: see `test_dequantise_returns_the_bin_left_edge`."""
        extent, n_bins = 300.0, 60
        step = extent / n_bins
        vals = np.linspace(0, extent - 1e-6, 97)
        back = dequantise(quantise(vals, n_bins, extent), n_bins, extent)
        self.assertTrue(np.all(np.abs(back - vals) <= step + 1e-9))


class TestEncoding(unittest.TestCase):

    def setUp(self):
        self.boxes = np.array([[[10.0, 20.0, 40.0, 60.0],
                                [100.0, 50.0, 150.0, 90.0]]])

    def test_one_bit_per_coordinate_run(self):
        z = boxes_to_latents(self.boxes, bins_x=8, bins_y=8)
        per_obj = bits_per_object(8, 8)
        self.assertEqual(z.shape, (1, 2 * per_obj))
        for o in range(2):
            block = z[0, o * per_obj:(o + 1) * per_obj]
            for r in range(4):
                run = block[r * 8:(r + 1) * 8]
                self.assertEqual(run.sum(), 1, f"object {o} run {r}")

    def test_latent_is_binary(self):
        z = boxes_to_latents(self.boxes, bins_x=8, bins_y=8)
        self.assertTrue(set(np.unique(z)).issubset({0, 1}))

    def test_padding_encodes_to_zeros(self):
        boxes = np.array([[[10.0, 20.0, 40.0, 60.0],
                           [0.0, 0.0, 0.0, 0.0]]])
        per_obj = bits_per_object(8, 8)
        z = boxes_to_latents(boxes, bins_x=8, bins_y=8)
        self.assertEqual(z[0, per_obj:].sum(), 0)

    def test_padding_round_trips_to_zeros(self):
        boxes = np.array([[[10.0, 20.0, 40.0, 60.0],
                           [0.0, 0.0, 0.0, 0.0]]])
        z = boxes_to_latents(boxes, bins_x=8, bins_y=8)
        back = latents_to_boxes(z, 2, bins_x=8, bins_y=8)
        np.testing.assert_array_equal(back[0, 1], np.zeros(4))
        self.assertTrue(np.any(back[0, 0] > 0))

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            boxes_to_latents(np.zeros((4, 3)))

    def test_decode_rejects_mismatched_width(self):
        z = boxes_to_latents(self.boxes, bins_x=8, bins_y=8)
        with self.assertRaises(ValueError):
            latents_to_boxes(z, 3, bins_x=8, bins_y=8)


class TestHamming(unittest.TestCase):

    def test_one_bin_move_flips_exactly_two_bits(self):
        """The reason for one-hot rather than a binary code."""
        step = CANVAS_W / 60.0
        a = np.array([[[10.0, 20.0, 40.0, 60.0]]])
        b = a.copy()
        b[0, 0, 0] += step                      # nudge x1 into the next bin
        za = boxes_to_latents(a, bins_x=60, bins_y=40)
        zb = boxes_to_latents(b, bins_x=60, bins_y=40)
        self.assertEqual(int((za ^ zb).sum()), 2)

    def test_identical_boxes_give_identical_latents(self):
        a = np.array([[[10.0, 20.0, 40.0, 60.0]]])
        za = boxes_to_latents(a, bins_x=16, bins_y=16)
        zb = boxes_to_latents(a.copy(), bins_x=16, bins_y=16)
        np.testing.assert_array_equal(za, zb)


class TestRoundTripError(unittest.TestCase):

    def test_error_is_bounded_by_bin_size(self):
        rng = np.random.RandomState(0)
        boxes = np.stack([
            rng.uniform(0, CANVAS_W, size=(20, 3)),
            rng.uniform(0, CANVAS_H, size=(20, 3)),
            rng.uniform(0, CANVAS_W, size=(20, 3)),
            rng.uniform(0, CANVAS_H, size=(20, 3)),
        ], axis=-1)
        mse = round_trip_error(boxes, bins_x=60, bins_y=40)
        # Four coordinates, each at most one WHOLE bin out, because the
        # dequantiser returns the bin's left edge to match `common/decode.py`.
        worst = 2 * (CANVAS_W / 60) ** 2 + 2 * (CANVAS_H / 40) ** 2
        self.assertLessEqual(mse, worst + 1e-6)

    def test_finer_bins_reduce_error(self):
        rng = np.random.RandomState(1)
        boxes = rng.uniform(0, 150, size=(30, 2, 4))
        coarse = round_trip_error(boxes, bins_x=8, bins_y=8)
        fine = round_trip_error(boxes, bins_x=64, bins_y=64)
        self.assertLess(fine, coarse)


class TestBuildExport(unittest.TestCase):

    def _moving_box(self, n=12):
        boxes = np.zeros((n, 2, 4), dtype=np.float32)
        for t in range(n):
            boxes[t, 0] = [10 + 8 * t, 20, 50 + 8 * t, 70]
            boxes[t, 1] = [200, 100 + 4 * t, 240, 150 + 4 * t]
        return boxes

    def test_export_loads_through_the_common_reader(self):
        from tools.planner.common.export import load
        boxes = self._moving_box()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "oracle.npz")
            info = build_export(boxes, path, bins_x=16, bins_y=16)
            ex = load(path)
            self.assertEqual(len(ex), len(boxes))
            self.assertEqual(ex.n_bits, info["n_bits"])
            self.assertEqual(ex.gt_boxes.shape, boxes.shape)
            # Every latent came from this table, so nothing should fall back.
            ex.boxes_for(ex.latents[3])
            self.assertEqual(ex.fallback_count, 0)

    def test_decoded_boxes_are_close_to_ground_truth(self):
        boxes = self._moving_box()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "oracle.npz")
            build_export(boxes, path, bins_x=60, bins_y=40)
            data = np.load(path)
            d2 = data["decoded_boxes"] - data["gt_boxes"]
            self.assertLess(float((d2 * d2).sum(axis=-1).mean()), 30.0)

    def test_distinct_latents_track_distinct_positions(self):
        boxes = self._moving_box()
        with tempfile.TemporaryDirectory() as d:
            info = build_export(boxes, os.path.join(d, "o.npz"),
                                bins_x=60, bins_y=40)
            self.assertEqual(info["distinct_latents"], len(boxes))

    def test_transitions_are_deduplicated(self):
        boxes = np.repeat(self._moving_box(4), 3, axis=0)   # each frame x3
        with tempfile.TemporaryDirectory() as d:
            info = build_export(boxes, os.path.join(d, "o.npz"),
                                bins_x=16, bins_y=16)
            self.assertLess(info["transitions"], len(boxes) - 1)

    def test_single_state_produces_no_transitions(self):
        boxes = self._moving_box(1)
        with tempfile.TemporaryDirectory() as d:
            info = build_export(boxes, os.path.join(d, "o.npz"),
                                bins_x=8, bins_y=8)
            self.assertEqual(info["transitions"], 0)



class TestBboxIoU(unittest.TestCase):
    """IoU is reported next to bbox_mse because MSE is not scale-invariant."""

    def _iou(self, pred, gt, **kw):
        from tools.planner.common.metrics import bbox_iou
        return bbox_iou(np.asarray(pred, dtype=float),
                        np.asarray(gt, dtype=float), **kw)

    def test_identical_boxes_score_one(self):
        b = [[[10.0, 10.0, 20.0, 20.0]]]
        r = self._iou(b, b, mapping=[0])
        self.assertAlmostEqual(r["mean_iou"], 1.0)

    def test_disjoint_boxes_score_zero(self):
        p = [[[0.0, 0.0, 10.0, 10.0]]]
        g = [[[50.0, 50.0, 60.0, 60.0]]]
        self.assertAlmostEqual(self._iou(p, g, mapping=[0])["mean_iou"], 0.0)

    def test_half_overlap(self):
        # Two 10x10 boxes sharing a 5x10 strip: intersection 50, union 150.
        p = [[[0.0, 0.0, 10.0, 10.0]]]
        g = [[[5.0, 0.0, 15.0, 10.0]]]
        self.assertAlmostEqual(self._iou(p, g, mapping=[0])["mean_iou"],
                               50.0 / 150.0, places=6)

    def test_padded_slot_scores_zero_not_one(self):
        """An all-zero box on both sides has zero union and must not count
        as a perfect overlap.

        Since 2026-08-30 `bbox_iou` excludes absent ground truth rather than
        scoring it as a miss, so a window that is nothing BUT padding has no
        scoreable frame at all and reports None -- the same "no data" signal
        `bbox_mse` now gives. Either way it is not 1.0, which is what this
        test exists to prevent.
        """
        p = [[[0.0, 0.0, 0.0, 0.0]]]
        r = self._iou(p, p, mapping=[0])
        self.assertIsNone(r["mean_iou"])

    def test_scale_invariance_is_the_point(self):
        """The property MSE lacks: the same relative error on a big and a
        small box scores the same."""
        small_p = [[[0.0, 0.0, 10.0, 10.0]]]
        small_g = [[[1.0, 0.0, 11.0, 10.0]]]
        big_p = [[[0.0, 0.0, 100.0, 100.0]]]
        big_g = [[[10.0, 0.0, 110.0, 100.0]]]
        a = self._iou(small_p, small_g, mapping=[0])["mean_iou"]
        b = self._iou(big_p, big_g, mapping=[0])["mean_iou"]
        self.assertAlmostEqual(a, b, places=6)

    def test_unmatched_slot_is_skipped(self):
        p = [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]]
        g = [[[0.0, 0.0, 10.0, 10.0], [90.0, 90.0, 99.0, 99.0]]]
        r = self._iou(p, g, mapping=[0, -1])
        self.assertAlmostEqual(r["mean_iou"], 1.0)

    def test_counts_frames_over_the_half_threshold(self):
        p = [[[0.0, 0.0, 10.0, 10.0]], [[0.0, 0.0, 10.0, 10.0]]]
        g = [[[0.0, 0.0, 10.0, 10.0]], [[50.0, 50.0, 60.0, 60.0]]]
        r = self._iou(p, g, mapping=[0])
        self.assertEqual(r["frames_above_0.5"], 1)
        self.assertEqual(r["n_frames"], 2)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self._iou([[[0.0, 0, 1, 1]]], [[[0.0, 0, 1, 1], [0, 0, 1, 1]]])

    def test_score_window_reports_both_metrics(self):
        from tools.planner.common.metrics import score_window
        pred = np.array([[[0.0, 0, 10, 10]], [[5.0, 0, 15, 10]]])
        gt = np.array([[[0.0, 0, 10, 10]], [[0.0, 0, 10, 10]]])
        base = np.array([[[0.0, 0, 10, 10]], [[0.0, 0, 10, 10]]])
        r = score_window(pred, gt, base, matching="fixed")
        self.assertIn("planner_iou", r)
        self.assertIn("baseline_iou", r)
        self.assertAlmostEqual(r["baseline_iou"]["mean_iou"], 1.0)
        self.assertLess(r["planner_iou"]["mean_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
