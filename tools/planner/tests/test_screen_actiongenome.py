#!/usr/bin/env python3
"""Screening Action Genome: runs, gaps, and what counts as a relation.

Two things here are easy to get wrong and both were got wrong once.

**A run is not a clip.** Counting maximal runs gives a larger number than
counting clips that contain at least one usable run, because one clip can
contribute several. The first hand count of this corpus reported 741 runs and
the tool reports 553 clips. The clip count is the honest one, because a clip is
what gets baked.

**"No relation" is written down explicitly.** The annotation format records
`['unsure']` and `['not_contacting']` rather than omitting the field, so
counting every populated field as a relation inflates the density of every frame
in the corpus.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from tools.video import screen_actiongenome as ag  # noqa: E402


def rec(attention="['looking_at']", spatial="['in_front_of']",
        contacting="['holding']", visible=True):
    return {"attention_relationship": attention,
            "spatial_relationship": spatial,
            "contacting_relationship": contacting,
            "visible": visible, "bbox": (0, 0, 10, 10), "class": "cup"}


class TestNonTrivialRelations(unittest.TestCase):

    def test_three_real_relations_count_three(self):
        self.assertEqual(ag.non_trivial_relations(rec()), 3)

    def test_unsure_is_not_a_relation(self):
        self.assertEqual(
            ag.non_trivial_relations(rec(attention="['unsure']")), 2)

    def test_not_contacting_is_not_a_relation(self):
        self.assertEqual(
            ag.non_trivial_relations(rec(contacting="['not_contacting']")), 2)

    def test_an_empty_list_is_not_a_relation(self):
        self.assertEqual(ag.non_trivial_relations(rec(spatial="[]")), 2)

    def test_a_frame_of_absences_carries_none(self):
        empty = rec(attention="['unsure']", spatial="[]",
                    contacting="['not_contacting']")
        self.assertEqual(ag.non_trivial_relations(empty), 0)


class TestRuns(unittest.TestCase):

    def test_consecutive_frames_are_one_run(self):
        self.assertEqual(ag.runs_within([1, 2, 3, 4], 1), [4])

    def test_a_gap_beyond_tolerance_splits_the_run(self):
        self.assertEqual(ag.runs_within([1, 2, 20, 21, 22], 4), [2, 3])

    def test_tolerance_admits_the_gap_it_names(self):
        self.assertEqual(ag.runs_within([1, 5, 9], 4), [3])
        self.assertEqual(ag.runs_within([1, 5, 9], 3), [1, 1, 1])

    def test_a_single_frame_is_a_run_of_one(self):
        self.assertEqual(ag.runs_within([7], 6), [1])

    def test_no_frames_is_no_runs(self):
        self.assertEqual(ag.runs_within([], 6), [])

    def test_order_does_not_matter(self):
        self.assertEqual(ag.runs_within([4, 1, 3, 2], 1), [4])


class TestScreen(unittest.TestCase):

    def clip(self, frames):
        return dict((f, [rec()]) for f in frames)

    def test_a_clip_qualifies_on_its_longest_run_not_its_frame_count(self):
        """Forty frames in scattered pairs must not qualify."""
        scattered = self.clip([i * 50 for i in range(40)])
        out = ag.screen({"a.mp4": scattered}, max_gap=6, min_run=8)
        self.assertEqual(out["clips"], 1)
        self.assertEqual(out["qualifying_clips"], 0)

    def test_one_dense_run_qualifies(self):
        dense = self.clip(list(range(0, 40, 4)))
        out = ag.screen({"a.mp4": dense}, max_gap=6, min_run=8)
        self.assertEqual(out["qualifying_clips"], 1)

    def test_a_clip_with_several_runs_still_counts_once(self):
        """The distinction that produced 741 instead of 553."""
        frames = list(range(0, 40, 4)) + list(range(400, 440, 4))
        out = ag.screen({"a.mp4": self.clip(frames)}, max_gap=6, min_run=8)
        self.assertEqual(out["qualifying_clips"], 1)

    def test_tightening_the_tolerance_cannot_add_clips(self):
        by_clip = {"a.mp4": self.clip(list(range(0, 40, 5))),
                   "b.mp4": self.clip(list(range(0, 40, 2)))}
        loose = ag.screen(by_clip, max_gap=6, min_run=8)["qualifying_clips"]
        tight = ag.screen(by_clip, max_gap=2, min_run=8)["qualifying_clips"]
        self.assertLessEqual(tight, loose)

    def test_invisible_objects_are_not_counted(self):
        by_clip = {"a.mp4": dict((f, [rec(), rec(visible=False)])
                                 for f in range(10))}
        out = ag.screen(by_clip, max_gap=1, min_run=2)
        self.assertEqual(out["median_objects_per_frame"], 1)


class TestFigure(unittest.TestCase):

    def test_the_sweep_is_rendered_rather_than_one_number(self):
        """One number invites reading the corpus size as intrinsic."""
        svg = ag.render_svg({"min_run": 8, "clips": 9601, "median_gap": 16,
                             "sweep": [(2, 34), (6, 553), (12, 2126)]})
        self.assertIn("553", svg)
        self.assertIn("2126", svg)
        self.assertTrue(svg.startswith("<svg"))

    def test_no_sweep_means_no_figure(self):
        self.assertIsNone(ag.render_svg({"min_run": 8, "clips": 1,
                                         "median_gap": 1}))


if __name__ == "__main__":
    unittest.main()
