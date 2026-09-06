"""Tests for the export liveness survey.

The failure this guards against is specific and has already cost real work: an
export whose encoder emits one code for every frame scores badly in every
downstream metric, and it scores badly for a reason that has nothing to do with
the property under test. Four separate pieces of work each rediscovered the
same four dead files by hand before this module existed.
"""

import json
import os
import shutil
import tempfile
import unittest
import xml.dom.minidom

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools.planner import liveness


def write_export(path, latents, **extra):
    arrays = {"latents": np.asarray(latents)}
    arrays.update(extra)
    np.savez_compressed(path, **arrays)
    return path


class TestDistinctLatents(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_an_all_zero_export_reads_as_one_distinct_code(self):
        """The exact shape of the four dead files found on 2026-09-05."""
        path = write_export(os.path.join(self.dir, "dead.npz"),
                            np.zeros((105, 200), dtype="int8"))
        facts = liveness.export_facts(path)
        self.assertEqual(facts["distinct"], 1)
        self.assertEqual(facts["ones"], 0)
        self.assertTrue(facts["dead"])

    def test_a_constant_but_nonzero_export_is_also_dead(self):
        """Deadness is one distinct code, not the code being zero.

        A model that emits the same non-zero code everywhere is exactly as
        useless to a planner, and reading deadness off the bit count alone
        would miss it.
        """
        latents = np.ones((40, 16), dtype="int8")
        path = write_export(os.path.join(self.dir, "const.npz"), latents)
        facts = liveness.export_facts(path)
        self.assertEqual(facts["distinct"], 1)
        self.assertEqual(facts["ones"], 640)
        self.assertTrue(facts["dead"])

    def test_a_live_export_counts_its_codes(self):
        latents = np.array([[0, 0], [0, 1], [1, 0], [0, 1]], dtype="int8")
        path = write_export(os.path.join(self.dir, "live.npz"), latents)
        facts = liveness.export_facts(path)
        self.assertEqual(facts["distinct"], 3)
        self.assertEqual(facts["frames"], 4)
        self.assertEqual(facts["bits"], 2)
        self.assertFalse(facts["dead"])

    def test_an_unreadable_file_is_reported_and_does_not_raise(self):
        """A survey over a directory must not die on one bad file."""
        path = os.path.join(self.dir, "broken.npz")
        with open(path, "w") as handle:
            handle.write("not an npz")
        facts = liveness.export_facts(path)
        self.assertIsNone(facts["distinct"])
        self.assertIn("error", facts)

    def test_an_export_without_latents_says_so(self):
        path = os.path.join(self.dir, "nolatents.npz")
        np.savez_compressed(path, gt_boxes=np.zeros((3, 2, 4)))
        facts = liveness.export_facts(path)
        self.assertIsNone(facts["distinct"])


class TestSurvey(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        write_export(os.path.join(self.dir, "a_dead.npz"),
                     np.zeros((10, 8), dtype="int8"))
        write_export(os.path.join(self.dir, "b_live.npz"),
                     np.eye(6, dtype="int8"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_survey_finds_every_export_and_sorts_dead_first(self):
        """The dead ones are the point, so they must not be buried."""
        rows = liveness.survey(self.dir)
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["dead"])
        self.assertFalse(rows[1]["dead"])

    def test_an_empty_directory_gives_no_rows_rather_than_failing(self):
        empty = tempfile.mkdtemp()
        try:
            self.assertEqual(liveness.survey(empty), [])
        finally:
            shutil.rmtree(empty, ignore_errors=True)


class TestValLossCorrelation(unittest.TestCase):
    """The 2026-09-05 finding: a degenerate loss plateau means a dead latent."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.csv = os.path.join(self.dir, "train.csv")
        with open(self.csv, "w") as handle:
            handle.write("u,p,best_val,epochs\n")
            handle.write("10,10,0.5243,3000\n")
            handle.write("20,10,0.524995,3000\n")
            handle.write("40,10,0.224848,3000\n")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_shapes_at_the_plateau_are_separated_from_the_rest(self):
        rows = liveness.read_train_csv(self.csv)
        plateau = liveness.plateau_shapes(rows)
        self.assertIn((10, 10), plateau)
        self.assertIn((20, 10), plateau)
        self.assertNotIn((40, 10), plateau)

    def test_a_single_row_is_never_called_a_plateau(self):
        """One value cannot agree with itself. Two runs are the minimum."""
        rows = liveness.read_train_csv(self.csv)
        self.assertEqual(liveness.plateau_shapes(rows[:1]), set())

    def test_the_plateau_value_is_measured_not_hardcoded(self):
        """0.5245 is what this dataset produced, not a constant of nature.

        Hard-coding it would silently mislabel a different dataset, which is
        the mistake this project keeps making with numbers.
        """
        rows = [{"u": 1, "p": 1, "best_val": 0.31, "epochs": 3000},
                {"u": 2, "p": 2, "best_val": 0.310004, "epochs": 3000},
                {"u": 3, "p": 3, "best_val": 0.19, "epochs": 3000}]
        plateau = liveness.plateau_shapes(rows)
        self.assertEqual(plateau, {(1, 1), (2, 2)})


class TestFigure(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        write_export(os.path.join(self.dir, "dead.npz"),
                     np.zeros((10, 8), dtype="int8"))
        write_export(os.path.join(self.dir, "live.npz"),
                     np.eye(6, dtype="int8"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_the_figure_is_well_formed_xml(self):
        """A raw angle bracket has shipped an unopenable figure three times."""
        svg = liveness.render_svg(liveness.survey(self.dir))
        xml.dom.minidom.parseString(svg)

    def test_the_figure_names_the_dead_count_in_words(self):
        svg = liveness.render_svg(liveness.survey(self.dir))
        self.assertIn("1 of 2", svg)

    def test_the_figure_survives_having_nothing_to_draw(self):
        empty = tempfile.mkdtemp()
        try:
            svg = liveness.render_svg(liveness.survey(empty))
            xml.dom.minidom.parseString(svg)
        finally:
            shutil.rmtree(empty, ignore_errors=True)


class TestCommandLine(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = tempfile.mkdtemp()
        write_export(os.path.join(self.dir, "dead.npz"),
                     np.zeros((10, 8), dtype="int8"))
        write_export(os.path.join(self.dir, "live.npz"),
                     np.eye(6, dtype="int8"))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)
        shutil.rmtree(self.out, ignore_errors=True)

    def test_it_writes_a_report_and_a_figure(self):
        code = liveness.main(["--exports", self.dir, "--out-dir", self.out])
        self.assertEqual(code, 0)
        report = os.path.join(self.out, "liveness.json")
        self.assertTrue(os.path.exists(report))
        with open(report) as handle:
            data = json.load(handle)
        self.assertEqual(data["dead"], 1)
        self.assertEqual(data["total"], 2)
        xml.dom.minidom.parse(os.path.join(self.out, "liveness.svg"))

    def test_strict_fails_when_anything_is_dead(self):
        """So a gate can refuse to score a grid that is mostly collapse."""
        code = liveness.main(["--exports", self.dir, "--out-dir", self.out,
                              "--strict"])
        self.assertEqual(code, 1)

    def test_a_missing_directory_reports_rather_than_crashing(self):
        code = liveness.main(["--exports", os.path.join(self.dir, "nope"),
                              "--out-dir", self.out])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
