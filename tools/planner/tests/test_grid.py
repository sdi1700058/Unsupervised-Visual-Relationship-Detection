"""The grid enumerates cells and refuses the impossible ones.

Four empty states stay distinct: never run, cannot run, ran and collapsed, ran
and scored. Merging them is how a dead export became a planning result. A cell
that cannot exist is not a cell that scored badly, and a collapsed run is an
absent measurement rather than a poor one.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import grid


def a_plan():
    return {
        "datasets_for_grid": [
            {"name": "vidvrd", "frames": True},
            {"name": "vidor", "frames": False},
        ],
        "methods_for_grid": ["interpolation", "M7"],
        "combinations": [],
        "units": [], "milestones": [], "claims": [],
    }


class TestEnumerate(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        with open(os.path.join(self.root, "there.json"), "w") as handle:
            handle.write("{}")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_every_dataset_by_method_by_source_appears(self):
        cells = grid.enumerate_cells(a_plan(), root=self.root)
        self.assertEqual(len(cells), 2 * 2 * 2)

    def test_a_trained_cell_without_frames_cannot_run(self):
        """A cell that cannot exist is not a cell that scored badly."""
        cells = grid.enumerate_cells(a_plan(), root=self.root)
        trained = [c for c in cells
                   if c["dataset"] == "vidor" and c["source"] == "trained"]
        self.assertTrue(trained)
        for cell in trained:
            self.assertEqual(cell["state"], grid.CANNOT_RUN)
            self.assertIn("frames", cell["reason"])

    def test_an_oracle_cell_without_frames_can_still_run(self):
        """The oracle needs boxes and never frames. This is the whole unlock."""
        cells = grid.enumerate_cells(a_plan(), root=self.root)
        oracle = [c for c in cells
                  if c["dataset"] == "vidor" and c["source"] == "oracle"]
        self.assertTrue(oracle)
        for cell in oracle:
            self.assertEqual(cell["state"], grid.NEVER_RUN)

    def test_a_cell_with_evidence_reads_as_scored(self):
        plan = a_plan()
        plan["combinations"] = [{"dataset": "vidvrd", "method": "M7",
                                 "source": "oracle",
                                 "evidence": "there.json"}]
        cells = grid.enumerate_cells(plan, root=self.root)
        scored = [c for c in cells if c["state"] == grid.SCORED]
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored[0]["method"], "M7")

    def test_a_declared_cell_whose_evidence_is_gone_is_not_scored(self):
        """A declaration is not a result. The file has to be there."""
        plan = a_plan()
        plan["combinations"] = [{"dataset": "vidvrd", "method": "M7",
                                 "source": "oracle", "evidence": "gone.json"}]
        cells = grid.enumerate_cells(plan, root=self.root)
        self.assertEqual([c for c in cells if c["state"] == grid.SCORED], [])

    def test_the_four_states_are_distinct(self):
        states = set([grid.NEVER_RUN, grid.CANNOT_RUN, grid.COLLAPSED,
                      grid.SCORED])
        self.assertEqual(len(states), 4)

    def test_missing_returns_only_the_runnable_gaps(self):
        cells = grid.enumerate_cells(a_plan(), root=self.root)
        for cell in grid.missing(cells):
            self.assertEqual(cell["state"], grid.NEVER_RUN)

    def test_an_empty_plan_enumerates_nothing_rather_than_raising(self):
        plan = {"datasets_for_grid": [], "methods_for_grid": [],
                "combinations": []}
        self.assertEqual(grid.enumerate_cells(plan, root=self.root), [])


class TestCommands(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_a_command_is_produced_for_each_missing_cell(self):
        cells = grid.enumerate_cells(a_plan(), root=self.root)
        gaps = grid.missing(cells)
        commands = grid.plan_commands(gaps)
        self.assertEqual(len(commands), len(gaps))
        for command in commands:
            self.assertTrue(command.strip())

    def test_no_command_is_produced_for_a_cell_that_cannot_run(self):
        cells = [c for c in grid.enumerate_cells(a_plan(), root=self.root)
                 if c["state"] == grid.CANNOT_RUN]
        self.assertTrue(cells)
        self.assertEqual(grid.plan_commands(cells), [])

    def test_a_command_names_its_dataset_and_its_method(self):
        cells = grid.enumerate_cells(a_plan(), root=self.root)
        for cell, command in zip(grid.missing(cells),
                                 grid.plan_commands(grid.missing(cells))):
            self.assertIn(cell["dataset"], command)


class TestSummary(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_the_summary_counts_each_state_separately(self):
        cells = grid.enumerate_cells(a_plan(), root=self.root)
        counts = grid.summarise(cells)
        self.assertEqual(counts[grid.CANNOT_RUN], 2)
        self.assertEqual(counts[grid.NEVER_RUN], 6)


class TestMain(unittest.TestCase):

    def test_it_is_a_dry_run_by_default_and_returns_zero(self):
        """Nothing is submitted until the author has seen the list."""
        code = grid.main(["--plan", "notes/WORKPLAN.json"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
