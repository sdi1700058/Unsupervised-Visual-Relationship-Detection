"""Twenty paired combinations, and no way to inflate the count.

A trained number cannot be read on its own. Whether an `mse_ratio` of 9.77 is
bad depends entirely on what the oracle scored on the same clips, the same
windows and the same frames. The oracle gives the ceiling, the trained run
gives the actual, and the gap between them is the finding.

Counting halves would let cheap oracle runs read as progress, and the oracle
needs no cluster at all. SPEC V38 already records that planner error must be
compared paired, and the E1 error of 2026-09-01 was an unpaired comparison that
reported the opposite of what its own data said.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import workplan


class TestPairs(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        for name in ("o.json", "t.json"):
            with open(os.path.join(self.root, name), "w") as handle:
                handle.write("{}")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def plan(self, combos):
        return {"units": [], "milestones": [], "claims": [],
                "combinations": combos}

    def test_both_halves_present_makes_one_pair(self):
        plan = self.plan([
            {"dataset": "vidvrd", "method": "M7", "source": "oracle",
             "evidence": "o.json"},
            {"dataset": "vidvrd", "method": "M7", "source": "trained",
             "evidence": "t.json"}])
        self.assertEqual(workplan.paired_combinations(plan, root=self.root),
                         ["vidvrd x M7"])

    def test_an_oracle_half_alone_is_not_a_pair(self):
        """The whole point. Cheap oracle runs must not inflate the count."""
        plan = self.plan([
            {"dataset": "vidvrd", "method": "M7", "source": "oracle",
             "evidence": "o.json"}])
        self.assertEqual(workplan.paired_combinations(plan, root=self.root), [])
        self.assertEqual(workplan.screening_only(plan, root=self.root),
                         ["vidvrd x M7 (oracle)"])

    def test_a_trained_half_alone_is_not_a_pair_either(self):
        plan = self.plan([
            {"dataset": "vidvrd", "method": "M7", "source": "trained",
             "evidence": "t.json"}])
        self.assertEqual(workplan.paired_combinations(plan, root=self.root), [])

    def test_a_half_whose_evidence_is_missing_does_not_count(self):
        plan = self.plan([
            {"dataset": "vidvrd", "method": "M7", "source": "oracle",
             "evidence": "o.json"},
            {"dataset": "vidvrd", "method": "M7", "source": "trained",
             "evidence": "gone.json"}])
        self.assertEqual(workplan.paired_combinations(plan, root=self.root), [])

    def test_a_declared_half_with_no_evidence_path_does_not_count(self):
        plan = self.plan([
            {"dataset": "vidvrd", "method": "M7", "source": "oracle",
             "evidence": "o.json"},
            {"dataset": "vidvrd", "method": "M7", "source": "trained"}])
        self.assertEqual(workplan.paired_combinations(plan, root=self.root), [])

    def test_pairs_on_different_datasets_are_different_pairs(self):
        plan = self.plan([
            {"dataset": "vidvrd", "method": "M7", "source": "oracle",
             "evidence": "o.json"},
            {"dataset": "vidvrd", "method": "M7", "source": "trained",
             "evidence": "t.json"},
            {"dataset": "vidor", "method": "M7", "source": "oracle",
             "evidence": "o.json"},
            {"dataset": "vidor", "method": "M7", "source": "trained",
             "evidence": "t.json"}])
        self.assertEqual(workplan.paired_combinations(plan, root=self.root),
                         ["vidor x M7", "vidvrd x M7"])

    def test_a_source_defaults_to_oracle_when_absent(self):
        """Every combination recorded before the source field existed was an
        oracle run, so absent must not mean trained."""
        plan = self.plan([
            {"dataset": "vidvrd", "method": "M7", "evidence": "o.json"}])
        self.assertEqual(workplan.screening_only(plan, root=self.root),
                         ["vidvrd x M7 (oracle)"])


class TestUnitsFraction(unittest.TestCase):

    def test_a_milestone_reports_how_far_its_units_have_come(self):
        plan = {"units": [{"id": "a", "milestone": "M3", "state": "built"},
                          {"id": "b", "milestone": "M3",
                           "state": "not_started"}],
                "milestones": [{"id": "M3", "title": "t", "target": 20,
                                "counter": "parked"}],
                "claims": [], "combinations": []}
        rows = workplan.milestone_progress(plan)
        self.assertIn("units_fraction", rows[0])
        self.assertGreater(rows[0]["units_fraction"], 0.0)
        self.assertLess(rows[0]["units_fraction"], 1.0)

    def test_a_milestone_with_no_units_reports_zero_rather_than_dividing(self):
        plan = {"units": [],
                "milestones": [{"id": "M5", "title": "t", "target": 1,
                                "counter": "parked"}],
                "claims": [], "combinations": []}
        rows = workplan.milestone_progress(plan)
        self.assertEqual(rows[0]["units_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
