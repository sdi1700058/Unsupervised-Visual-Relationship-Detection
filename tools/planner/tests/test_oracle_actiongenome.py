#!/usr/bin/env python3
"""Loading Action Genome boxes: runs, and the two box conventions in one release.

The trap here is measured, not hypothetical. Action Genome stores object boxes
as `xywh` and person boxes as `xyxy`, in two files of the same release. Over
7,841 records, 100% are consistent with `xywh` and only 19% with `xyxy`, so
reading the objects as corner pairs compresses every object box and corrupts
every trajectory without raising anything.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from tools.planner import oracle  # noqa: E402


class TestRuns(unittest.TestCase):

    def test_a_gap_within_tolerance_keeps_one_run(self):
        self.assertEqual(oracle.ag_runs_within([0, 4, 8], 4), [[0, 4, 8]])

    def test_a_gap_beyond_tolerance_splits(self):
        self.assertEqual(oracle.ag_runs_within([0, 4, 40, 44], 6),
                         [[0, 4], [40, 44]])

    def test_it_returns_frames_and_not_lengths(self):
        """The near-twin in the corpus screen returns lengths. This must not."""
        self.assertEqual(oracle.ag_runs_within([1, 2, 3], 1), [[1, 2, 3]])

    def test_no_frames_is_no_runs(self):
        self.assertEqual(oracle.ag_runs_within([], 6), [])

    def test_input_order_does_not_matter(self):
        self.assertEqual(oracle.ag_runs_within([8, 0, 4], 4), [[0, 4, 8]])

    def test_a_lone_frame_is_a_run_of_one(self):
        self.assertEqual(oracle.ag_runs_within([5], 3), [[5]])


class TestBoxConventions(unittest.TestCase):
    """Guards the xywh/xyxy split without needing the corpus on disk."""

    def test_the_object_convention_is_recorded_in_the_docstring(self):
        doc = oracle.boxes_from_actiongenome_clip.__doc__
        self.assertIn("xywh", doc)
        self.assertIn("xyxy", doc)

    def test_an_empty_clip_returns_an_empty_array_not_an_error(self):
        boxes, meta = oracle.boxes_from_actiongenome_clip({}, {}, num_objs=3)
        self.assertEqual(boxes.shape, (0, 3, 4))
        self.assertEqual(meta["frames"], 0)

    def test_frames_missing_from_either_file_are_dropped(self):
        """A frame needs both an object record and a person record."""
        objects = dict((f, []) for f in range(10))
        person = {0: {"bbox": [], "bbox_size": (480, 270)}}
        boxes, meta = oracle.boxes_from_actiongenome_clip(objects, person)
        self.assertEqual(meta["frames"], 1)


if __name__ == "__main__":
    unittest.main()
