"""The unit ladder, and the one state a tool may never set.

A unit passes through states that each report a checkable fact, up to
`evidence_produced`. `accepted` is a judgement about whether the work is
sufficient, so it belongs to the author. This distinction exists because
`PROGRESS.md` was once found carrying two headings the assistant had written on
its own authority, closing work items nobody had reviewed.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import workplan


def a_plan(state="not_started"):
    return {"units": [{"id": "U-one", "title": "a unit", "state": state,
                       "produces": "code"}],
            "milestones": [], "claims": [], "combinations": []}


class TestLadder(unittest.TestCase):

    def test_the_ladder_runs_from_not_started_to_accepted(self):
        self.assertEqual(workplan.LADDER[0], "not_started")
        self.assertEqual(workplan.LADDER[-1], "accepted")
        for state in ("designed", "built", "measured", "illustrated",
                      "evidence_produced"):
            self.assertIn(state, workplan.LADDER)

    def test_a_state_can_be_set(self):
        plan = a_plan()
        unit = workplan.set_state(plan, "U-one", "built")
        self.assertEqual(unit["state"], "built")

    def test_accepted_is_refused(self):
        """A tool that could set this would close its own work item."""
        plan = a_plan()
        try:
            workplan.set_state(plan, "U-one", "accepted")
        except ValueError as exc:
            self.assertIn("author", str(exc).lower())
        else:
            self.fail("set_state accepted 'accepted'")

    def test_an_unknown_state_is_refused(self):
        plan = a_plan()
        self.assertRaises(ValueError, workplan.set_state, plan, "U-one", "done")

    def test_an_unknown_unit_is_refused(self):
        plan = a_plan()
        self.assertRaises(ValueError, workplan.set_state, plan, "U-nope",
                          "built")

    def test_the_fraction_rises_along_the_ladder(self):
        self.assertEqual(workplan.ladder_fraction({"state": "not_started"}),
                         0.0)
        self.assertEqual(workplan.ladder_fraction({"state": "accepted"}), 1.0)
        self.assertLess(workplan.ladder_fraction({"state": "built"}),
                        workplan.ladder_fraction({"state": "measured"}))

    def test_an_unknown_state_scores_zero_rather_than_raising(self):
        """The plan is edited by hand, so one bad value must not break the
        board that reads it."""
        self.assertEqual(workplan.ladder_fraction({"state": "banana"}), 0.0)

    def test_a_unit_with_no_state_scores_zero(self):
        self.assertEqual(workplan.ladder_fraction({}), 0.0)

    def test_the_legacy_in_progress_state_still_scores(self):
        """Units recorded before the ladder existed must stay readable."""
        self.assertGreaterEqual(workplan.ladder_fraction(
            {"state": "in_progress"}), 0.0)


class TestSave(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "plan.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_saved_plan_reads_back_identically(self):
        plan = a_plan("measured")
        workplan.save(plan, self.path)
        with open(self.path) as handle:
            again = json.load(handle)
        self.assertEqual(again["units"][0]["state"], "measured")

    def test_saving_does_not_lose_other_keys(self):
        plan = a_plan()
        plan["questions"] = [{"id": "Q1", "question": "why"}]
        workplan.save(plan, self.path)
        with open(self.path) as handle:
            again = json.load(handle)
        self.assertEqual(again["questions"][0]["id"], "Q1")


if __name__ == "__main__":
    unittest.main()
