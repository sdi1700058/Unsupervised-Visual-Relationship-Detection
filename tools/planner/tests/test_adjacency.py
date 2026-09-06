"""A transition is one step, or it is not a transition.

`build_transitions` paired frame `i` with frame `i+1` in the loaded array and
compared only the video id, never the frame number. Both video loaders skip
frames in silence -- a missing jpg, an empty annotation -- so every skip turned
into an "adjacent" pair.

Measured on the production configuration, 30fps, mo3, p8, 60 videos: **45 of
4,335 emitted transitions span 16 to 271 real frames**, up to nine seconds, as
a single action step. At 3fps every transition spans ten. FOSAE's action model
assumes one step, so the assumption was being violated without a word.

Decided 2026-09-06: **count always, refuse only under `STRICT_ADJACENCY=1`.**
Existing runs reproduce by default so the measurements stay comparable, new
runs can be clean, and the export records which it was.

`latplan/util/adjacency.py` imports nothing from `latplan`, so it loads from
file here without dragging in TensorFlow.
"""

import importlib.util
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))

_spec = importlib.util.spec_from_file_location(
    "_adjacency", os.path.join(ROOT, "latplan", "util", "adjacency.py"))
adjacency = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adjacency)


class TestParsing(unittest.TestCase):

    def test_a_frame_id_gives_its_video_and_its_number(self):
        self.assertEqual(adjacency.parse_frame_id("ILSVRC2015_train_005/000012"),
                         ("ILSVRC2015_train_005", 12))

    def test_a_video_id_containing_a_slash_keeps_everything_before_the_last(self):
        """Action Genome writes `<vid>.mp4/<NNNNNN>.png`."""
        self.assertEqual(adjacency.parse_frame_id("001YG.mp4/000030.png"),
                         ("001YG.mp4", 30))

    def test_an_unparsable_number_gives_none_rather_than_zero(self):
        """Zero would make every such frame look adjacent to frame one."""
        self.assertEqual(adjacency.parse_frame_id("vid/abc"), ("vid", None))

    def test_an_id_with_no_slash_is_all_video_and_no_number(self):
        self.assertEqual(adjacency.parse_frame_id("vid"), ("vid", None))


class TestPairs(unittest.TestCase):

    def test_consecutive_frames_pair(self):
        ids = ["v/000000", "v/000001", "v/000002"]
        pairs, stats = adjacency.sequential_pairs(ids)
        self.assertEqual(pairs, [(0, 1), (1, 2)])
        self.assertEqual(stats["non_adjacent"], 0)

    def test_a_video_boundary_is_never_paired(self):
        ids = ["a/000000", "a/000001", "b/000000"]
        pairs, _ = adjacency.sequential_pairs(ids)
        self.assertEqual(pairs, [(0, 1)])

    def test_a_gap_is_counted_and_still_paired_by_default(self):
        """The default keeps behaviour identical and stops being silent."""
        ids = ["v/000000", "v/000030"]
        pairs, stats = adjacency.sequential_pairs(ids)
        self.assertEqual(pairs, [(0, 1)])
        self.assertEqual(stats["non_adjacent"], 1)
        self.assertEqual(stats["max_gap"], 30)

    def test_strict_refuses_the_gap(self):
        ids = ["v/000000", "v/000030", "v/000031"]
        pairs, stats = adjacency.sequential_pairs(ids, strict=True)
        self.assertEqual(pairs, [(1, 2)])
        self.assertEqual(stats["non_adjacent"], 1)
        self.assertEqual(stats["refused"], 1)

    def test_nothing_is_refused_when_not_strict(self):
        ids = ["v/000000", "v/000030"]
        _, stats = adjacency.sequential_pairs(ids)
        self.assertEqual(stats["refused"], 0)

    def test_the_stats_name_the_mode_so_an_export_records_it(self):
        _, loose = adjacency.sequential_pairs(["v/0", "v/1"])
        _, tight = adjacency.sequential_pairs(["v/0", "v/1"], strict=True)
        self.assertFalse(loose["strict"])
        self.assertTrue(tight["strict"])

    def test_an_unnumbered_frame_counts_as_unknown_not_as_adjacent(self):
        """Silence about a gap is the defect; guessing is the same defect."""
        ids = ["v/abc", "v/def"]
        pairs, stats = adjacency.sequential_pairs(ids)
        self.assertEqual(pairs, [(0, 1)])
        self.assertEqual(stats["unknown_gap"], 1)
        self.assertEqual(stats["non_adjacent"], 0)

    def test_strict_keeps_a_pair_whose_gap_cannot_be_known(self):
        """Refusing on ignorance would silently drop whole datasets."""
        pairs, _ = adjacency.sequential_pairs(["v/abc", "v/def"], strict=True)
        self.assertEqual(pairs, [(0, 1)])

    def test_the_totals_add_up(self):
        ids = ["a/000000", "a/000001", "a/000010", "b/000000", "b/000001"]
        pairs, stats = adjacency.sequential_pairs(ids)
        self.assertEqual(stats["pairs"], len(pairs))
        self.assertEqual(stats["pairs"], 3)
        self.assertEqual(stats["boundaries"], 1)
        self.assertEqual(stats["non_adjacent"], 1)
        self.assertEqual(stats["max_gap"], 9)

    def test_one_frame_gives_no_pairs_and_does_not_raise(self):
        pairs, stats = adjacency.sequential_pairs(["v/000000"])
        self.assertEqual(pairs, [])
        self.assertEqual(stats["pairs"], 0)

    def test_no_frames_gives_no_pairs(self):
        self.assertEqual(adjacency.sequential_pairs([])[0], [])


class TestTheEnvironmentSwitch(unittest.TestCase):

    def setUp(self):
        self._was = os.environ.get("STRICT_ADJACENCY")

    def tearDown(self):
        if self._was is None:
            os.environ.pop("STRICT_ADJACENCY", None)
        else:
            os.environ["STRICT_ADJACENCY"] = self._was

    def test_it_is_off_unless_asked(self):
        os.environ.pop("STRICT_ADJACENCY", None)
        self.assertFalse(adjacency.strict_from_env())

    def test_one_turns_it_on(self):
        os.environ["STRICT_ADJACENCY"] = "1"
        self.assertTrue(adjacency.strict_from_env())

    def test_zero_leaves_it_off(self):
        os.environ["STRICT_ADJACENCY"] = "0"
        self.assertFalse(adjacency.strict_from_env())


class TestTheReport(unittest.TestCase):
    """What gets printed, so the loss is visible without reading JSON."""

    def test_a_clean_load_says_so(self):
        _, stats = adjacency.sequential_pairs(["v/000000", "v/000001"])
        self.assertIn("all", adjacency.describe(stats).lower())

    def test_a_lossy_load_names_the_count_and_the_worst_gap(self):
        _, stats = adjacency.sequential_pairs(["v/000000", "v/000271"])
        text = adjacency.describe(stats)
        self.assertIn("1", text)
        self.assertIn("271", text)


if __name__ == "__main__":
    unittest.main()
