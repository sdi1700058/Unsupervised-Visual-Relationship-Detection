#!/usr/bin/env python3
"""Tests for M4, latent distance against the number of frames between states.

M4 turns `SPEC.md` V35 from a diagnosis of one clip into a metric. The tests
below fix the three things V35 never pinned down: that the ratio carries a
spread and not only a median, that the curve is checked for saturation, and
that a pair of frames never crosses a clip boundary.

    .venv-local/bin/python -m unittest \\
        tools.planner.tests.test_m4_temporal
"""

import os
import sys
import unittest
import xml.dom.minidom

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))


def _preserving(n=40, bits=64):
    """One new bit per frame, so frames k apart are k transitions apart.

    This is what a positional code does when every frame moves a box across at
    least one bin boundary.
    """
    z = np.zeros((n, bits), dtype=np.int8)
    for i in range(1, n):
        z[i] = z[i - 1]
        z[i, i % bits] = 1
    return z


def _compressed(n=40, bits=64, every=3):
    """The code changes only every `every` frames, so it compresses time."""
    z = np.zeros((n, bits), dtype=np.int8)
    for i in range(1, n):
        z[i] = z[i - 1]
        if i % every == 0:
            z[i, (i // every) % bits] = 1
    return z


def _saturating(n=40, bits=64, until=10):
    """The code advances for `until` frames and then never changes again.

    Beyond that point no frame gap produces a longer latent distance, which is
    exactly the failure that stops a planner separating a long horizon from a
    short one.
    """
    z = np.zeros((n, bits), dtype=np.int8)
    for i in range(1, n):
        z[i] = z[i - 1]
        if i < until:
            z[i, i % bits] = 1
    return z


def _ids(clip, n, step=1):
    return np.asarray(["%s/%06d" % (clip, i * step) for i in range(n)])


class TestRatio(unittest.TestCase):

    def test_a_code_that_preserves_time_scores_one(self):
        from tools.planner.m4_temporal import temporal_distance

        r = temporal_distance(_preserving(), max_gap=12)
        self.assertAlmostEqual(r["ratio_median"], 1.0, places=6)
        self.assertGreater(r["spearman"], 0.95)
        self.assertGreater(r["pearson"], 0.95)

    def test_a_compressed_code_scores_below_one(self):
        from tools.planner.m4_temporal import temporal_distance

        r = temporal_distance(_compressed(every=3), max_gap=12)
        self.assertLess(r["ratio_median"], 0.5)
        self.assertGreater(r["ratio_median"], 0.15)

    def test_compression_is_ordered_by_severity(self):
        from tools.planner.m4_temporal import temporal_distance

        mild = temporal_distance(_compressed(every=2), max_gap=12)
        harsh = temporal_distance(_compressed(every=5), max_gap=12)
        self.assertGreater(mild["ratio_median"], harsh["ratio_median"])

    def test_the_spread_is_reported_beside_the_median(self):
        """V35 quoted medians only, and a median cannot separate a code that
        compresses everywhere from one exact on half its pairs and dead on the
        rest."""
        from tools.planner.m4_temporal import temporal_distance

        r = temporal_distance(_compressed(every=3), max_gap=12)
        for key in ("ratio_q25", "ratio_q75", "ratio_iqr", "ratio_mean",
                    "ratio_sd", "exact_fraction"):
            self.assertIn(key, r)
        self.assertLessEqual(r["ratio_q25"], r["ratio_median"])
        self.assertLessEqual(r["ratio_median"], r["ratio_q75"])
        self.assertAlmostEqual(r["ratio_iqr"], r["ratio_q75"] - r["ratio_q25"],
                               places=9)


class TestSaturation(unittest.TestCase):

    def test_a_preserving_code_never_saturates(self):
        from tools.planner.m4_temporal import temporal_distance

        r = temporal_distance(_preserving(n=60), max_gap=20)
        self.assertIsNone(r["saturation_gap"])
        self.assertEqual(r["monotone_to"], 20)

    def test_a_frozen_code_saturates_and_the_gap_is_reported(self):
        from tools.planner.m4_temporal import temporal_distance

        r = temporal_distance(_saturating(n=60, until=10), max_gap=20)
        self.assertIsNotNone(r["saturation_gap"])
        self.assertLess(r["saturation_gap"], 20)

    def test_the_median_curve_is_returned_for_every_gap(self):
        from tools.planner.m4_temporal import temporal_distance

        r = temporal_distance(_preserving(n=60), max_gap=8)
        self.assertEqual(sorted(r["per_gap"].keys()), list(range(1, 9)))
        self.assertEqual(r["per_gap"][5]["median"], 5.0)
        self.assertIn("q25", r["per_gap"][5])
        self.assertIn("n", r["per_gap"][5])


class TestClipBoundaries(unittest.TestCase):

    def test_no_pair_crosses_a_clip_boundary(self):
        """A multi-clip export concatenates clips. A pair spanning the join
        compares two frames that no amount of time separates."""
        from tools.planner.m4_temporal import temporal_distance

        one = _preserving(n=30, bits=64)
        two = _preserving(n=30, bits=64)[:, ::-1].copy()
        two[:, 0] = 1              # a marker bit, so the clips share no code
        z = np.concatenate([one, two])
        ids = np.concatenate([_ids("a", 30), _ids("b", 30)])

        joined = temporal_distance(z, frame_ids=ids, max_gap=10)
        alone = temporal_distance(one, frame_ids=_ids("a", 30), max_gap=10)

        self.assertEqual(joined["n_clips"], 2)
        self.assertEqual(joined["n_pairs"], 2 * alone["n_pairs"])
        self.assertAlmostEqual(joined["ratio_median"], 1.0, places=6)
        self.assertEqual(joined["unreachable"], 0)

    def test_an_export_without_frame_ids_is_one_clip(self):
        from tools.planner.m4_temporal import temporal_distance

        r = temporal_distance(_preserving(n=30), max_gap=10)
        self.assertEqual(r["n_clips"], 1)


class TestFrameGap(unittest.TestCase):

    def test_the_gap_comes_from_the_frame_ids_not_the_row_index(self):
        """Sub-sampled frames sit two apart in time and one apart in the
        array. Using the index would report a code as twice as faithful as it
        is."""
        from tools.planner.m4_temporal import temporal_distance

        z = _preserving(n=30)
        r = temporal_distance(z, frame_ids=_ids("a", 30, step=2), max_gap=12)
        self.assertAlmostEqual(r["ratio_median"], 0.5, places=6)

    def test_a_frame_id_without_a_number_falls_back_to_the_index(self):
        from tools.planner.m4_temporal import temporal_distance

        ids = np.asarray(["clip/first"] * 4 + ["clip/second"] * 26)
        r = temporal_distance(_preserving(n=30), frame_ids=ids, max_gap=8)
        self.assertGreater(r["n_pairs"], 0)


class TestDegenerate(unittest.TestCase):

    def test_a_dead_latent_is_undefined_not_zero(self):
        """One distinct code means no distance exists. None, not 0.0 (V29)."""
        from tools.planner.m4_temporal import temporal_distance, verdict

        r = temporal_distance(np.zeros((40, 32), dtype=np.int8), max_gap=10)
        self.assertIsNone(r["ratio_median"])
        self.assertIsNone(r["spearman"])
        self.assertEqual(r["n_distinct"], 1)
        self.assertTrue(verdict(r).startswith("SILENT"))

    def test_a_clip_shorter_than_two_frames_is_skipped(self):
        from tools.planner.m4_temporal import temporal_distance

        r = temporal_distance(_preserving(n=1), max_gap=10)
        self.assertEqual(r["n_pairs"], 0)


class TestVerdict(unittest.TestCase):

    def test_silence_is_checked_before_the_ratio(self):
        """A ratio computed over a code with no temporal signal measures
        noise, and must not be quoted as compression."""
        from tools.planner.m4_temporal import verdict

        r = {"n_distinct": 40, "spearman": 0.10, "ratio_median": 0.99,
             "ratio_iqr": 0.05, "saturation_gap": None}
        self.assertTrue(verdict(r).startswith("SILENT"))

    def test_a_wide_spread_forbids_quoting_the_median_alone(self):
        from tools.planner.m4_temporal import verdict

        r = {"n_distinct": 40, "spearman": 0.90, "ratio_median": 0.80,
             "ratio_q25": 0.40, "ratio_q75": 1.00, "ratio_iqr": 0.60,
             "saturation_gap": None}
        self.assertIn("quartiles", verdict(r).lower())

    def test_a_faithful_code_reads_as_preserved(self):
        from tools.planner.m4_temporal import verdict

        r = {"n_distinct": 40, "spearman": 0.99, "ratio_median": 1.00,
             "ratio_q25": 1.00, "ratio_q75": 1.00, "ratio_iqr": 0.0,
             "saturation_gap": None}
        self.assertIn("PRESERVED", verdict(r))

    def test_a_compressed_code_reads_as_compressed(self):
        from tools.planner.m4_temporal import verdict

        r = {"n_distinct": 40, "spearman": 0.90, "ratio_median": 0.60,
             "ratio_q25": 0.55, "ratio_q75": 0.70, "ratio_iqr": 0.15,
             "saturation_gap": None}
        self.assertIn("COMPRESSED", verdict(r))


class TestHamming(unittest.TestCase):

    def test_hamming_is_reported_as_a_second_view(self):
        """The transition graph can collapse. Hamming distance needs no graph,
        so it still reports when the graph says nothing."""
        from tools.planner.m4_temporal import temporal_distance

        r = temporal_distance(_preserving(n=40), max_gap=12)
        self.assertGreater(r["hamming_spearman"], 0.95)


class TestFigure(unittest.TestCase):

    def test_the_figure_parses_as_xml(self):
        """This project has shipped an unopenable figure three times by
        writing a raw angle bracket into SVG text."""
        import tempfile

        from tools.planner.m4_temporal import temporal_distance, write_svg

        rows = [("a < b", temporal_distance(_preserving(n=40), max_gap=10)),
                ("compressed", temporal_distance(_compressed(), max_gap=10))]
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as fh:
            path = fh.name
        try:
            write_svg(rows, path)
            xml.dom.minidom.parse(path)
            with open(path) as fh:
                body = fh.read()
            self.assertIn("&lt;", body)
        finally:
            os.unlink(path)

    def test_a_dead_row_does_not_break_the_figure(self):
        import tempfile

        from tools.planner.m4_temporal import temporal_distance, write_svg

        dead = temporal_distance(np.zeros((20, 8), dtype=np.int8), max_gap=5)
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as fh:
            path = fh.name
        try:
            write_svg([("dead", dead)], path)
            xml.dom.minidom.parse(path)
        finally:
            os.unlink(path)


class TestCli(unittest.TestCase):

    def test_a_label_can_be_attached_to_a_path(self):
        from tools.planner.m4_temporal import split_label

        self.assertEqual(split_label("oracle=eval/probe/batch"),
                         ("oracle", "eval/probe/batch"))
        self.assertEqual(split_label("eval/exports/x.npz"),
                         ("x", "eval/exports/x.npz"))


if __name__ == "__main__":
    unittest.main()
