"""A unit is done when three artefacts exist, and not before.

The three are code with a test, an evidence file, and a figure. This guards
against the failure that has cost this project most: a number stated in a
document that no file on disk can reproduce. The headline pair `floor_ratio`
1.56 and "oracle closer on 19 of 19" sat in four documents for five days with
no summary carrying the column that produced it.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import workplan

SVG = '<svg xmlns="http://www.w3.org/2000/svg"><text>x &lt; y</text></svg>'


class TestVerify(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, rel, text):
        path = os.path.join(self.root, rel)
        directory = os.path.dirname(path)
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        with open(path, "w") as handle:
            handle.write(text)
        return rel

    def plan_with(self, evidence, touches=None, produces="measurement"):
        return {"units": [{"id": "U-one", "title": "a unit",
                           "state": "measured", "produces": produces,
                           "evidence": evidence,
                           "touches": touches or []}],
                "milestones": [], "claims": [], "combinations": []}

    def test_a_unit_with_all_three_artefacts_verifies(self):
        self.write("eval/result.json", '{"n": 3}')
        self.write("eval/result.svg", SVG)
        self.write("tools/planner/tests/test_thing.py", "# a test")
        plan = self.plan_with(["eval/result.json", "eval/result.svg"],
                              ["tools/planner/tests/test_thing.py"],
                              produces="code")
        out = workplan.verify_unit(plan, "U-one", root=self.root)
        self.assertTrue(out["ok"], out["checks"])

    def test_a_missing_evidence_file_fails(self):
        self.write("eval/result.svg", SVG)
        plan = self.plan_with(["eval/gone.json", "eval/result.svg"])
        out = workplan.verify_unit(plan, "U-one", root=self.root)
        self.assertFalse(out["ok"])

    def test_a_unit_with_no_figure_fails(self):
        """Every result needs something to look at. This is a requirement."""
        self.write("eval/result.json", '{"n": 3}')
        plan = self.plan_with(["eval/result.json"])
        out = workplan.verify_unit(plan, "U-one", root=self.root)
        self.assertFalse(out["ok"])
        failed = [c["name"] for c in out["checks"] if not c["ok"]]
        self.assertIn("figure", failed)

    def test_a_figure_that_is_not_valid_xml_fails(self):
        """A raw angle bracket has shipped an unopenable figure three times."""
        self.write("eval/result.json", "{}")
        self.write("eval/bad.svg", "<svg><text>a < b</text></svg>")
        plan = self.plan_with(["eval/result.json", "eval/bad.svg"])
        out = workplan.verify_unit(plan, "U-one", root=self.root)
        self.assertFalse(out["ok"])
        failed = [c["name"] for c in out["checks"] if not c["ok"]]
        self.assertIn("figure", failed)

    def test_a_unit_with_no_evidence_at_all_fails(self):
        plan = self.plan_with([])
        out = workplan.verify_unit(plan, "U-one", root=self.root)
        self.assertFalse(out["ok"])

    def test_a_png_counts_as_a_figure_and_is_not_parsed(self):
        """matplotlib output is a figure too, and it is not XML."""
        self.write("eval/result.json", "{}")
        self.write("eval/plot.png", "not really a png")
        plan = self.plan_with(["eval/result.json", "eval/plot.png"])
        out = workplan.verify_unit(plan, "U-one", root=self.root)
        figure = [c for c in out["checks"] if c["name"] == "figure"][0]
        self.assertTrue(figure["ok"])

    def test_a_measurement_unit_does_not_owe_a_test_file(self):
        """Only a unit producing code owes a test; a measurement owes data."""
        self.write("eval/result.json", "{}")
        self.write("eval/result.svg", SVG)
        plan = self.plan_with(["eval/result.json", "eval/result.svg"])
        out = workplan.verify_unit(plan, "U-one", root=self.root)
        self.assertTrue(out["ok"], out["checks"])

    def test_a_code_unit_without_a_test_file_fails(self):
        self.write("eval/result.json", "{}")
        self.write("eval/result.svg", SVG)
        plan = self.plan_with(["eval/result.json", "eval/result.svg"],
                              touches=["tools/thing.py"], produces="code")
        out = workplan.verify_unit(plan, "U-one", root=self.root)
        self.assertFalse(out["ok"])
        failed = [c["name"] for c in out["checks"] if not c["ok"]]
        self.assertIn("test", failed)


    def test_only_a_measurement_owes_a_figure(self):
        """"Every result needs something to look at" is about results.

        Found 2026-09-06: the figure was demanded of every unit, so the eight
        that wrote documents or built tooling all failed for lacking a picture
        they were never going to have.
        """
        self.write("notes/docs/THING.md", "# a document")
        plan = {"units": [{"id": "U-doc", "title": "wrote a document",
                           "state": "accepted", "produces": "document",
                           "evidence": ["notes/docs/THING.md"],
                           "touches": []}],
                "milestones": [], "claims": [], "combinations": []}
        out = workplan.verify_unit(plan, "U-doc", root=self.root)
        self.assertTrue(out["ok"], out["checks"])

    def test_a_measurement_still_owes_a_figure(self):
        self.write("eval/r.json", "{}")
        plan = self.plan_with(["eval/r.json"])
        out = workplan.verify_unit(plan, "U-one", root=self.root)
        self.assertFalse(out["ok"])

    def test_an_unknown_unit_raises(self):
        plan = self.plan_with([])
        self.assertRaises(ValueError, workplan.verify_unit, plan, "U-nope")



class TestAcceptedIsVerifiedToo(unittest.TestCase):
    """An accepted unit is the one whose claims are most load-bearing.

    Found 2026-09-06: `verify` with no --unit targeted measured, illustrated
    and evidence_produced only. Nine accepted units were never re-checked, so
    if the evidence behind an accepted result disappeared nothing would notice.
    """

    def test_accepted_is_among_the_states_verified_by_default(self):
        self.assertIn("accepted", workplan.VERIFY_STATES)

    def test_the_unfinished_states_are_not_verified(self):
        """A unit that has not claimed a result owes nothing yet."""
        self.assertNotIn("not_started", workplan.VERIFY_STATES)
        self.assertNotIn("designed", workplan.VERIFY_STATES)


if __name__ == "__main__":
    unittest.main()
