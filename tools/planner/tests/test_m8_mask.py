#!/usr/bin/env python3
"""Tests for M8, scoring a plan against masks rather than boxes.

Two things are checked here that nothing else in the suite can check.

1. **Agreement.** When a mask happens to be a rectangle, the mask metric must
   return what the box metric returns. That is the only way to know the new
   pipeline is measuring the same quantity as the old one, and it is testable
   today on the box datasets already on disk.
2. **The vanishing object.** The mask code drops any object smaller than half
   a grid cell. `oracle._code_width` documents the same class of bug on the box
   side, where an object in the top bin decoded as absent. The mask code must
   count these rather than silently lose them.
"""

import os
import shutil
import sys
import tempfile
import unittest
import xml.dom.minidom
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def square(height, width, r0, r1, c0, c1):
    """A filled rectangle mask covering rows [r0, r1) and columns [c0, c1)."""
    m = np.zeros((height, width), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


class TestMaskToBox(unittest.TestCase):
    """The conversion half: a tight box from a mask."""

    def test_tight_box_is_the_extent_of_the_set_pixels(self):
        from tools.planner.m8_mask import masks_to_boxes

        m = square(40, 60, 10, 20, 5, 25)
        box = masks_to_boxes(m)
        # Half-open, so the width is x2 - x1 and equals the covered columns.
        self.assertTrue(np.allclose(box, [5.0, 10.0, 25.0, 20.0]))

    def test_an_empty_mask_is_an_absent_object_not_a_box_at_the_origin(self):
        from tools.planner.m8_mask import masks_to_boxes

        box = masks_to_boxes(np.zeros((40, 60), dtype=bool))
        self.assertTrue(np.allclose(box, [0.0, 0.0, 0.0, 0.0]))

    def test_a_ragged_mask_still_gives_the_enclosing_box(self):
        from tools.planner.m8_mask import masks_to_boxes

        m = np.zeros((20, 20), dtype=bool)
        m[3, 4] = True
        m[9, 12] = True
        self.assertTrue(np.allclose(masks_to_boxes(m), [4.0, 3.0, 13.0, 10.0]))

    def test_it_maps_over_leading_axes(self):
        from tools.planner.m8_mask import masks_to_boxes

        masks = np.stack([
            np.stack([square(20, 20, 1, 5, 2, 6),
                      np.zeros((20, 20), dtype=bool)]),
            np.stack([square(20, 20, 0, 3, 0, 3),
                      square(20, 20, 10, 12, 10, 12)]),
        ])
        boxes = masks_to_boxes(masks)
        self.assertEqual(boxes.shape, (2, 2, 4))
        self.assertTrue(np.allclose(boxes[0, 0], [2, 1, 6, 5]))
        self.assertTrue(np.allclose(boxes[0, 1], [0, 0, 0, 0]))
        self.assertTrue(np.allclose(boxes[1, 1], [10, 10, 12, 12]))

    def test_box_to_mask_to_box_is_the_identity_on_integer_boxes(self):
        """The datasets on disk hold boxes, so this round trip must be exact."""
        from tools.planner.m8_mask import boxes_to_masks, masks_to_boxes

        rng = np.random.RandomState(0)
        boxes = np.zeros((30, 2, 4))
        boxes[:, :, 0] = rng.randint(0, 200, size=(30, 2))
        boxes[:, :, 1] = rng.randint(0, 120, size=(30, 2))
        boxes[:, :, 2] = boxes[:, :, 0] + rng.randint(1, 90, size=(30, 2))
        boxes[:, :, 3] = boxes[:, :, 1] + rng.randint(1, 70, size=(30, 2))

        back = masks_to_boxes(boxes_to_masks(boxes, height=200, width=300))
        self.assertTrue(np.array_equal(back, boxes))

    def test_an_absent_box_rasterises_to_an_empty_mask(self):
        from tools.planner.m8_mask import boxes_to_masks

        masks = boxes_to_masks(np.zeros((3, 2, 4)), height=20, width=20)
        self.assertEqual(masks.shape, (3, 2, 20, 20))
        self.assertFalse(masks.any())


class TestMaskIoU(unittest.TestCase):
    """The region metric."""

    def test_identical_masks_score_one(self):
        from tools.planner.m8_mask import mask_iou

        m = square(40, 40, 5, 25, 5, 25)
        self.assertAlmostEqual(mask_iou(m, m), 1.0)

    def test_disjoint_masks_score_zero(self):
        from tools.planner.m8_mask import mask_iou

        a = square(40, 40, 0, 10, 0, 10)
        b = square(40, 40, 20, 30, 20, 30)
        self.assertAlmostEqual(mask_iou(a, b), 0.0)

    def test_a_shifted_square_scores_the_analytic_value(self):
        """A square of side s shifted by d along one axis scores (s-d)/(s+d)."""
        from tools.planner.m8_mask import mask_iou

        s, d = 20, 4
        gt = square(60, 60, 10, 10 + s, 10, 10 + s)
        pred = square(60, 60, 10, 10 + s, 10 + d, 10 + s + d)
        self.assertAlmostEqual(mask_iou(pred, gt), (s - d) / float(s + d))

    def test_two_empty_masks_are_unscoreable_rather_than_perfect(self):
        """Union zero means there was nothing to score, not a perfect hit."""
        from tools.planner.m8_mask import mask_iou

        empty = np.zeros((10, 10), dtype=bool)
        self.assertIsNone(mask_iou(empty, empty))


class TestDiceIsNotASecondMetric(unittest.TestCase):
    """Jaccard and Dice are one measure, and must not be counted as two.

    Dice equals `2J / (1 + J)`, which is strictly increasing in J. The two
    therefore rank any set of predictions identically and neither can ever
    contradict the other. Dice is worth reporting, because the segmentation
    literature reports it and it makes this work comparable, but a milestone
    counting distinct evaluation methods must count the pair once.
    """

    def test_dice_is_exactly_the_monotone_transform_of_iou(self):
        from tools.planner.m8_mask import dice_from_iou, mask_dice, mask_iou

        rng = np.random.RandomState(11)
        for _ in range(25):
            a = rng.rand(24, 24) > 0.5
            b = rng.rand(24, 24) > 0.5
            j = mask_iou(a, b)
            self.assertAlmostEqual(mask_dice(a, b), 2 * j / (1 + j),
                                   places=12)
            self.assertAlmostEqual(mask_dice(a, b), dice_from_iou(j),
                                   places=12)

    def test_dice_and_iou_rank_the_same_predictions_in_the_same_order(self):
        """The operational consequence: the pair cannot disagree."""
        from tools.planner.m8_mask import mask_dice, mask_iou

        gt = square(60, 60, 10, 40, 10, 40)
        preds = [square(60, 60, 10 + d, 40 + d, 10, 40) for d in
                 (0, 3, 7, 12, 20, 29)]
        ious = [mask_iou(p, gt) for p in preds]
        dices = [mask_dice(p, gt) for p in preds]
        self.assertEqual(list(np.argsort(ious)), list(np.argsort(dices)))

    def test_both_refuse_an_empty_pair_rather_than_scoring_it_perfect(self):
        from tools.planner.m8_mask import dice_from_iou, mask_dice

        self.assertIsNone(mask_dice(np.zeros((8, 8), dtype=bool),
                                    np.zeros((8, 8), dtype=bool)))
        self.assertIsNone(dice_from_iou(None))


class TestBoundaryF(unittest.TestCase):
    """The boundary-sensitive metric, and what each metric is blind to."""

    def test_identical_masks_score_one(self):
        from tools.planner.m8_mask import boundary_f

        m = square(40, 40, 5, 25, 5, 25)
        self.assertAlmostEqual(boundary_f(m, m), 1.0)

    def test_disjoint_masks_score_zero(self):
        from tools.planner.m8_mask import boundary_f

        a = square(60, 60, 0, 10, 0, 10)
        b = square(60, 60, 40, 50, 40, 50)
        self.assertAlmostEqual(boundary_f(a, b), 0.0)

    def test_iou_is_blind_to_where_the_error_sits_and_boundary_f_is_not(self):
        """Two predictions with the SAME IoU and very different boundaries.

        This is the reason a second metric exists. Both predictions lose
        exactly 80 pixels of a 400-pixel square, so IoU cannot tell them
        apart. One loses a thin strip from each end and stays within the
        tolerance of the real outline; the other loses one thick slab and its
        edge sits 4 pixels away from where it should be.
        """
        from tools.planner.m8_mask import boundary_f, mask_iou

        gt = square(50, 50, 10, 30, 10, 30)
        thin_both_ends = square(50, 50, 12, 28, 10, 30)     # 16 x 20 = 320
        thick_one_end = square(50, 50, 14, 30, 10, 30)      # 16 x 20 = 320

        self.assertEqual(mask_iou(thin_both_ends, gt),
                         mask_iou(thick_one_end, gt))
        f_thin = boundary_f(thin_both_ends, gt, tol=2)
        f_thick = boundary_f(thick_one_end, gt, tol=2)
        self.assertGreater(f_thin, f_thick + 0.15)

    def test_iou_saturates_on_a_small_object_and_the_contour_does_not(self):
        """The second reason a boundary measure earns its place.

        The same 2-pixel displacement is a catastrophe for a 4-pixel object
        and a rounding error for a 40-pixel one, so IoU reports object size as
        much as it reports error. The contour measure at `tol=2` calls both of
        them the same, which is the other half of the trade: it is stable
        across scale and blind below its tolerance. Neither reading is right
        on its own, which is why both are reported.
        """
        from tools.planner.m8_mask import boundary_f, mask_iou

        small_gt = square(60, 60, 10, 14, 10, 14)
        small_pred = square(60, 60, 10, 14, 12, 16)
        big_gt = square(60, 60, 5, 45, 5, 45)
        big_pred = square(60, 60, 5, 45, 7, 47)

        self.assertAlmostEqual(mask_iou(small_pred, small_gt), 1.0 / 3.0)
        self.assertGreater(mask_iou(big_pred, big_gt), 0.9)
        self.assertAlmostEqual(boundary_f(small_pred, small_gt, tol=2), 1.0)
        self.assertAlmostEqual(boundary_f(big_pred, big_gt, tol=2), 1.0)

    def test_boundary_f_is_blind_to_error_below_its_tolerance(self):
        """The converse blindness, so neither metric is read as the truth."""
        from tools.planner.m8_mask import boundary_f, mask_iou

        gt = square(50, 50, 10, 30, 10, 30)
        eroded = square(50, 50, 11, 29, 11, 29)
        self.assertAlmostEqual(boundary_f(eroded, gt, tol=2), 1.0)
        self.assertLess(mask_iou(eroded, gt), 1.0)


class TestMaskLatents(unittest.TestCase):
    """The mask analogue of the box quantisation, and its floor."""

    def test_the_latent_is_one_bit_per_grid_cell_per_object(self):
        from tools.planner.m8_mask import mask_bits_per_object, masks_to_latents

        masks = np.zeros((4, 3, 20, 30), dtype=bool)
        z = masks_to_latents(masks, bins_x=6, bins_y=4)
        self.assertEqual(mask_bits_per_object(6, 4), 24)
        self.assertEqual(z.shape, (4, 3 * 24))

    def test_a_cell_aligned_mask_round_trips_exactly(self):
        from tools.planner.m8_mask import (latents_to_masks, mask_iou,
                                           masks_to_latents)

        masks = np.zeros((1, 1, 20, 30), dtype=bool)
        masks[0, 0, 5:15, 10:25] = True        # whole 5x5 cells at bins 6x4
        z = masks_to_latents(masks, bins_x=6, bins_y=4)
        back = latents_to_masks(z, 1, bins_x=6, bins_y=4, height=20, width=30)
        self.assertAlmostEqual(mask_iou(back[0, 0], masks[0, 0]), 1.0)

    def test_an_absent_object_stays_absent_through_the_code(self):
        """An all-zero block must not decode to a mask in the top-left cell."""
        from tools.planner.m8_mask import latents_to_masks, masks_to_latents

        masks = np.zeros((2, 2, 20, 30), dtype=bool)
        masks[:, 0, 5:15, 10:25] = True        # slot 1 is padding
        z = masks_to_latents(masks, bins_x=6, bins_y=4)
        self.assertEqual(int(z[:, 24:].sum()), 0)
        back = latents_to_masks(z, 2, bins_x=6, bins_y=4, height=20, width=30)
        self.assertFalse(back[:, 1].any())

    def test_an_object_smaller_than_half_a_cell_vanishes_and_is_counted(self):
        """The mask code has the failure `_code_width` fixed on the box side.

        A present object that decodes to nothing is a deleted object, and a
        floor that averaged over the survivors alone would hide it.
        """
        from tools.planner.m8_mask import round_trip_masks

        masks = np.zeros((1, 1, 20, 30), dtype=bool)
        masks[0, 0, 7:9, 12:14] = True         # 4 px inside one 5x5 cell
        r = round_trip_masks(masks, bins_x=6, bins_y=4)
        self.assertEqual(r["vanished"], 1)
        self.assertEqual(r["n_present"], 1)
        self.assertAlmostEqual(r["mean_iou"], 0.0)

    def test_the_floor_is_below_one_when_the_mask_does_not_fit_the_grid(self):
        from tools.planner.m8_mask import round_trip_masks

        masks = np.zeros((1, 1, 20, 30), dtype=bool)
        masks[0, 0, 3:16, 7:26] = True         # deliberately off the cell grid
        r = round_trip_masks(masks, bins_x=6, bins_y=4)
        self.assertLess(r["mean_iou"], 1.0)
        self.assertGreater(r["mean_iou"], 0.5)
        self.assertEqual(r["vanished"], 0)

    def test_the_floor_reports_nothing_when_no_object_is_present(self):
        from tools.planner.m8_mask import round_trip_masks

        r = round_trip_masks(np.zeros((3, 2, 20, 30), dtype=bool),
                             bins_x=6, bins_y=4)
        self.assertIsNone(r["mean_iou"])
        self.assertEqual(r["n_present"], 0)

    def test_the_mask_code_is_far_wider_than_the_box_code(self):
        """The cost of the representation, at the resolution actually used."""
        from tools.planner.m8_mask import mask_bits_per_object
        from tools.planner.oracle import (DEFAULT_BINS_X, DEFAULT_BINS_Y,
                                          bits_per_object)

        box = bits_per_object(DEFAULT_BINS_X, DEFAULT_BINS_Y)
        mask = mask_bits_per_object(DEFAULT_BINS_X, DEFAULT_BINS_Y)
        self.assertEqual(box, 200)
        self.assertEqual(mask, 2400)


class TestScoreMasks(unittest.TestCase):
    """Scoring a whole window, the way `bbox_mse` scores one."""

    def test_a_perfect_plan_scores_one(self):
        from tools.planner.m8_mask import score_masks

        gt = np.zeros((4, 2, 30, 30), dtype=bool)
        for t in range(4):
            gt[t, 0, 5 + t:15 + t, 5:15] = True
            gt[t, 1, 20:25, 20:25] = True
        r = score_masks(gt, gt)
        self.assertAlmostEqual(r["planner"]["mean_iou"], 1.0)
        self.assertAlmostEqual(r["planner"]["mean_dice"], 1.0)
        self.assertAlmostEqual(r["planner"]["mean_boundary_f"], 1.0)

    def test_the_reported_dice_is_derived_from_the_reported_iou(self):
        """Derived, not recomputed, so the two can never drift apart.

        A reader who sees two columns has to be able to find out that they are
        one measure. The window mean is a mean of per-pair transforms, so it
        is not `dice_from_iou(mean_iou)`, but every row it averages is.
        """
        from tools.planner.m8_mask import dice_from_iou, score_masks

        gt = np.zeros((2, 1, 30, 30), dtype=bool)
        pred = np.zeros((2, 1, 30, 30), dtype=bool)
        gt[:, 0, 5:20, 5:20] = True
        pred[:, 0, 8:23, 5:20] = True
        r = score_masks(pred, gt)["planner"]
        self.assertAlmostEqual(r["mean_dice"],
                               dice_from_iou(r["mean_iou"]), places=12)

    def test_absent_objects_are_excluded_rather_than_scored_as_misses(self):
        """The same exclusion `bbox_iou` applies, or the two are incomparable."""
        from tools.planner.m8_mask import score_masks

        gt = np.zeros((3, 2, 30, 30), dtype=bool)
        gt[:, 0, 5:15, 5:15] = True            # slot 1 never appears
        pred = gt.copy()
        pred[:, 1, 0:10, 0:10] = True          # a hallucination on empty truth
        r = score_masks(pred, gt, matching="fixed")
        self.assertAlmostEqual(r["planner"]["mean_iou"], 1.0)
        self.assertEqual(r["planner"]["skipped_absent"], 3)

    def test_a_window_with_nothing_present_reports_none_not_a_perfect_score(self):
        from tools.planner.m8_mask import score_masks

        empty = np.zeros((3, 2, 20, 20), dtype=bool)
        r = score_masks(empty, empty)
        self.assertIsNone(r["planner"]["mean_iou"])

    def test_slots_are_paired_over_the_window_so_a_swap_does_not_cost(self):
        from tools.planner.m8_mask import score_masks

        gt = np.zeros((3, 2, 30, 30), dtype=bool)
        gt[:, 0, 2:10, 2:10] = True
        gt[:, 1, 18:26, 18:26] = True
        swapped = gt[:, ::-1].copy()
        r = score_masks(swapped, gt)
        self.assertAlmostEqual(r["planner"]["mean_iou"], 1.0)
        self.assertEqual(r["planner"]["mapping"], [1, 0])

    def test_the_baseline_comparison_says_whether_the_plan_beat_a_line(self):
        from tools.planner.m8_mask import score_masks

        gt = np.zeros((3, 1, 30, 30), dtype=bool)
        pred = np.zeros((3, 1, 30, 30), dtype=bool)
        base = np.zeros((3, 1, 30, 30), dtype=bool)
        for t in range(3):
            gt[t, 0, 5:15, 5 + 2 * t:15 + 2 * t] = True
            pred[t, 0, 5:15, 5 + 2 * t:15 + 2 * t] = True     # exact
            base[t, 0, 5:15, 20:30] = True                    # far away
        r = score_masks(pred, gt, baseline_masks=base)
        self.assertTrue(r["beats_baseline"])
        self.assertGreater(r["planner"]["mean_iou"],
                           r["baseline"]["mean_iou"])


class TestAgreementWithTheBoxMetric(unittest.TestCase):
    """The correctness test: rectangles must score the same either way."""

    def test_mask_iou_equals_bbox_iou_when_the_masks_are_rectangles(self):
        from tools.planner.common.metrics import bbox_iou
        from tools.planner.m8_mask import boxes_to_masks, score_masks

        rng = np.random.RandomState(7)
        n_t, n_o = 6, 3
        gt = np.zeros((n_t, n_o, 4))
        gt[:, :, 0] = rng.randint(0, 200, size=(n_t, n_o))
        gt[:, :, 1] = rng.randint(0, 120, size=(n_t, n_o))
        gt[:, :, 2] = gt[:, :, 0] + rng.randint(5, 90, size=(n_t, n_o))
        gt[:, :, 3] = gt[:, :, 1] + rng.randint(5, 70, size=(n_t, n_o))
        pred = np.clip(gt + rng.randint(-15, 15, size=gt.shape), 0, None)
        pred[:, :, 2] = np.maximum(pred[:, :, 2], pred[:, :, 0] + 1)
        pred[:, :, 3] = np.maximum(pred[:, :, 3], pred[:, :, 1] + 1)

        box = bbox_iou(pred, gt, matching="fixed")["mean_iou"]
        mask = score_masks(boxes_to_masks(pred, 200, 300),
                           boxes_to_masks(gt, 200, 300),
                           matching="fixed")["planner"]["mean_iou"]
        self.assertAlmostEqual(box, mask, places=9)

    def test_the_agreement_check_reports_its_own_worst_case(self):
        from tools.planner.m8_mask import agreement_with_boxes

        rng = np.random.RandomState(3)
        gt = np.zeros((5, 2, 4))
        gt[:, :, 0] = rng.randint(0, 150, size=(5, 2))
        gt[:, :, 1] = rng.randint(0, 100, size=(5, 2))
        gt[:, :, 2] = gt[:, :, 0] + rng.randint(10, 80, size=(5, 2))
        gt[:, :, 3] = gt[:, :, 1] + rng.randint(10, 60, size=(5, 2))
        pred = gt + 7.0

        r = agreement_with_boxes(pred, gt, height=200, width=300)
        self.assertLess(r["max_abs_deviation"], 1e-9)
        self.assertEqual(r["n_pairs"], 10)

    def test_non_integer_boxes_agree_only_to_rasterisation_accuracy(self):
        """Honest about the one place the two cannot agree exactly.

        A real dataset box lands between pixels after the canvas rescale. A mask
        cannot hold half a pixel, so the two metrics differ by the area the
        rasteriser rounds. The check reports that number instead of hiding it.
        """
        from tools.planner.m8_mask import agreement_with_boxes

        gt = np.array([[[10.4, 20.6, 60.3, 70.9]]])
        pred = np.array([[[13.1, 22.2, 61.7, 74.4]]])
        r = agreement_with_boxes(pred, gt, height=200, width=300)
        self.assertGreater(r["max_abs_deviation"], 0.0)
        self.assertLess(r["max_abs_deviation"], 0.05)


class TestFigure(unittest.TestCase):
    """A figure that will not open is not a figure.

    This project has shipped an unopenable SVG three times, always because a
    raw `<` or `>` reached the text of a label.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="m8-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_written_figure_parses_as_xml(self):
        from tools.planner.m8_mask import write_figure

        report = {
            "floors": [
                {"name": "box one-hot 60x40 (iou < 1 here)", "bits": 200,
                 "iou": 0.83, "boundary_f": 0.62},
                {"name": "mask grid 60x40", "bits": 2400,
                 "iou": 0.91, "boundary_f": 0.75},
            ],
            "blindness": {"iou_thin": 0.8, "iou_thick": 0.8,
                          "bf_thin": 1.0, "bf_thick": 0.72},
            "agreement": {"max_abs_deviation": 3.2e-14, "n_pairs": 120,
                          "source": "synthetic rectangles"},
        }
        path = os.path.join(self.tmp, "m8_mask.svg")
        write_figure(report, path)
        xml.dom.minidom.parse(path)

    def test_a_long_caveat_stays_inside_the_canvas(self):
        """A caveat that runs off the edge is a caveat nobody reads."""
        from tools.planner.m8_mask import write_figure

        long_note = ("the raw canvas boxes of this dataset are already whole "
                     "pixels, because the canvas scaler rounds, so the "
                     "rasteriser costs nothing here and this row repeats the "
                     "one above, which is worth saying at length to be sure "
                     "the wrapping is exercised by this test") * 2
        report = {
            "subtitle": long_note,
            "floors": [{"name": "one", "bits": 200, "iou": 0.7,
                        "boundary_f": 0.7}],
            "blindness": {"iou_thin": 0.8, "iou_thick": 0.8,
                          "bf_thin": 1.0, "bf_thick": 0.7},
            "agreement": {"max_abs_deviation": 0.0, "n_pairs": 4,
                          "source": "synthetic"},
            "agreement_note": long_note,
        }
        path = os.path.join(self.tmp, "wrapped.svg")
        write_figure(report, path)
        doc = xml.dom.minidom.parse(path)
        svg = doc.documentElement
        canvas_h = float(svg.getAttribute("viewBox").split()[3])
        lowest = max(float(n.getAttribute("y"))
                     for n in doc.getElementsByTagName("text"))
        self.assertLessEqual(lowest, canvas_h)
        self.assertGreater(len(doc.getElementsByTagName("text")), 10)

    def test_a_label_carrying_angle_brackets_is_escaped(self):
        from tools.planner.m8_mask import write_figure

        report = {
            "floors": [{"name": "iou <= 1 & > 0", "bits": 8, "iou": 0.5,
                        "boundary_f": 0.5}],
            "blindness": {"iou_thin": 0.8, "iou_thick": 0.8,
                          "bf_thin": 1.0, "bf_thick": 0.7},
            "agreement": {"max_abs_deviation": 0.0, "n_pairs": 1,
                          "source": "a < b"},
        }
        path = os.path.join(self.tmp, "escaped.svg")
        write_figure(report, path)
        xml.dom.minidom.parse(path)
        with open(path) as f:
            body = f.read()
        self.assertNotIn("<= 1", body)
        self.assertIn("&lt;= 1", body)


if __name__ == "__main__":
    unittest.main()
