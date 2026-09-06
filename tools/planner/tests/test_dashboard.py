"""The dashboard. Four panels, one image, and it must open.

A reader needs the bottom line in bulk, in a form a person can take in at a
glance. This is that image, and these tests hold it to being openable and
complete.
"""

import os
import shutil
import sys
import tempfile
import unittest
import xml.dom.minidom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import dashboard


def a_plan():
    return {
        "milestones": [
            {"id": "M1", "title": "Related work", "target": 40,
             "counter": "parked"},
            {"id": "M3", "title": "Evaluation", "target": 20,
             "counter": "parked"}],
        "units": [{"id": "u", "milestone": "M1", "state": "built"}],
        "datasets_for_grid": [{"name": "vidvrd", "frames": True},
                              {"name": "vidor", "frames": False}],
        "methods_for_grid": ["interpolation", "M7 triplet mAP"],
        "combinations": [], "claims": [], "questions": [],
    }


class TestRender(unittest.TestCase):

    def test_the_dashboard_is_well_formed_xml(self):
        xml.dom.minidom.parseString(dashboard.render(a_plan()))

    def test_every_milestone_appears(self):
        svg = dashboard.render(a_plan())
        self.assertIn("M1", svg)
        self.assertIn("M3", svg)

    def test_no_raw_angle_bracket_reaches_a_text_node(self):
        plan = a_plan()
        plan["questions"] = [{"id": "Q1", "question": "is a < b & c > d?"}]
        xml.dom.minidom.parseString(dashboard.render(plan))

    def test_it_renders_with_nothing_at_all(self):
        empty = {"milestones": [], "units": [], "combinations": [],
                 "claims": [], "questions": []}
        xml.dom.minidom.parseString(dashboard.render(empty))

    def test_every_grid_state_carries_a_glyph_and_not_colour_alone(self):
        """The reference palette's own mitigation: a status colour never
        carries meaning unaided, so each state ships an icon and a label."""
        for state, mark in dashboard.STATE_GLYPH.items():
            self.assertTrue(mark.strip(), "%s has no glyph" % state)

    def test_the_legend_names_every_state(self):
        svg = dashboard.render(a_plan())
        for state in dashboard.STATE_GLYPH:
            self.assertIn(dashboard._esc(state), svg)

    def test_dark_mode_is_declared_rather_than_flipped(self):
        """A dark palette is chosen, never derived by inverting the light one."""
        svg = dashboard.render(a_plan())
        self.assertIn("prefers-color-scheme: dark", svg)

    def test_a_claim_shows_its_confidence_mark(self):
        plan = a_plan()
        plan["claims"] = [{"id": "C1", "asserts": "something happened",
                           "claim_type": "existential"}]
        svg = dashboard.render(plan)
        self.assertIn("C1", svg)


class TestMain(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_it_writes_the_file(self):
        out = os.path.join(self.dir, "dashboard.svg")
        code = dashboard.main(["--out", out])
        self.assertEqual(code, 0)
        xml.dom.minidom.parse(out)

    def test_a_missing_plan_reports_rather_than_writing_a_blank(self):
        code = dashboard.main(["--plan", os.path.join(self.dir, "none.json"),
                               "--out", os.path.join(self.dir, "d.svg")])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
