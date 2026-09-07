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

from tools.planner import box_geometry as oracle  # noqa: E402


class TestRuns(unittest.TestCase):

    def test_a_gap_within_tolerance_keeps_one_run(self):
        self.assertEqual(oracle.ag_runs_within([0, 4, 8], 4), [[0, 4, 8]])

    def test_a_gap_beyond_tolerance_splits(self):
        self.assertEqual(oracle.ag_runs_within([0, 4, 40, 44], 6),
                         [[0, 4], [40, 44]])

    def test_it_returns_frames_and_not_lengths(self):
        """The near-twin in the dataset screen returns lengths. This must not."""
        self.assertEqual(oracle.ag_runs_within([1, 2, 3], 1), [[1, 2, 3]])

    def test_no_frames_is_no_runs(self):
        self.assertEqual(oracle.ag_runs_within([], 6), [])

    def test_input_order_does_not_matter(self):
        self.assertEqual(oracle.ag_runs_within([8, 0, 4], 4), [[0, 4, 8]])

    def test_a_lone_frame_is_a_run_of_one(self):
        self.assertEqual(oracle.ag_runs_within([5], 3), [[5]])


def _has_pillow():
    """`load_canvas_scaler` reaches the loader, which imports PIL."""
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


class TestBoxConventions(unittest.TestCase):
    """Guards the xywh/xyxy split without needing the dataset on disk."""

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


class TestTheConventionsAsApplied(unittest.TestCase):
    """The conventions in the boxes, not in the prose that describes them.

    `TestBoxConventions` above reads the docstring, so it holds however the
    body behaves. Reading the object record as a corner pair -- the one
    failure this module calls corrupting every trajectory without raising
    anything -- left every test in this file green.
    """

    def _clip(self):
        objects, person = {}, {}
        for f in (0, 4, 8):
            objects[f] = [{"class": "cup", "visible": True,
                           "bbox": [10.0, 20.0, 30.0, 40.0]}]
            person[f] = {"bbox": [[100.0, 50.0, 160.0, 200.0]],
                         "bbox_size": (480, 270), "bbox_mode": "xyxy"}
        return objects, person

    def _loaded(self):
        objects, person = self._clip()
        boxes, meta = oracle.boxes_from_actiongenome_clip(objects, person,
                                                          num_objs=2)
        scale, _, _ = oracle.load_canvas_scaler()
        return boxes, meta, scale

    @unittest.skipUnless(_has_pillow(), "needs pillow (present in .venv-local)")
    def test_an_object_record_is_read_as_xywh(self):
        """`[10, 20, 30, 40]` covers x from 10 to 40, not from 10 to 30."""
        boxes, meta, scale = self._loaded()
        cup = meta["slots"].index("cup")
        right = tuple(float(v) for v in scale([10.0, 20.0, 40.0, 60.0],
                                              480, 270))
        wrong = tuple(float(v) for v in scale([10.0, 20.0, 30.0, 40.0],
                                              480, 270))
        self.assertEqual(tuple(boxes[0, cup]), right)
        # The guard is worth nothing if the two readings coincide.
        self.assertNotEqual(right, wrong)

    @unittest.skipUnless(_has_pillow(), "needs pillow (present in .venv-local)")
    def test_a_person_record_is_read_as_xyxy(self):
        boxes, meta, scale = self._loaded()
        person = meta["slots"].index("person")
        self.assertEqual(tuple(boxes[0, person]),
                         tuple(float(v) for v in
                               scale([100.0, 50.0, 160.0, 200.0], 480, 270)))

    @unittest.skipUnless(_has_pillow(), "needs pillow (present in .venv-local)")
    def test_the_person_is_the_last_slot(self):
        _, meta, _ = self._loaded()
        self.assertEqual(meta["slots"][-1], "person")


if __name__ == "__main__":
    unittest.main()
