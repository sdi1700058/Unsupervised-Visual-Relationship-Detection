"""Skills arrive at the decision point, and no invented skill passes.

A trigger table already existed in the standing instructions and was still
under-used, because that document is read once at the start of a session. By
the time a unit begins it is not re-scanned. The `skills` field appears in the
output of `workplan next`, which is read at the moment the unit is chosen.

Two failures on 2026-09-05 make the case. A cluster experiment was reported as
running when it had never been submitted, and models were exported before their
training had finished. Both are what verification-before-completion prevents.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import workplan


class TestSkillsFor(unittest.TestCase):

    def test_a_code_unit_gets_the_test_first_skill(self):
        got = workplan.skills_for({"produces": "code"})
        self.assertIn("superpowers:test-driven-development", got)

    def test_a_measurement_unit_gets_verification(self):
        """The skill that would have caught the two worst errors."""
        got = workplan.skills_for({"produces": "measurement"})
        self.assertIn("superpowers:verification-before-completion", got)

    def test_an_explicit_list_overrides_the_default(self):
        got = workplan.skills_for({"produces": "code",
                                   "skills": ["superpowers:brainstorming"]})
        self.assertEqual(got, ["superpowers:brainstorming"])

    def test_an_unknown_produces_gives_an_empty_list_rather_than_raising(self):
        self.assertEqual(workplan.skills_for({"produces": "banana"}), [])

    def test_a_unit_with_no_produces_gives_an_empty_list(self):
        self.assertEqual(workplan.skills_for({}), [])

    def test_every_default_skill_is_a_real_skill(self):
        """A default that named a skill which does not exist would teach the
        assistant to call something imaginary."""
        for produces, skills in workplan.SKILLS_BY_PRODUCES.items():
            for skill in skills:
                self.assertIn(skill, workplan.KNOWN_SKILLS,
                              "%s names %s" % (produces, skill))


class TestUnknownSkills(unittest.TestCase):

    def test_an_invented_skill_name_is_reported(self):
        """Inventing a skill name is the same class of error as inventing a
        url, and it is cheap to prevent."""
        plan = {"units": [{"id": "U-one", "produces": "code",
                           "skills": ["superpowers:does-not-exist"]}],
                "milestones": [], "claims": [], "combinations": []}
        self.assertEqual(workplan.unknown_skills(plan),
                         [("U-one", "superpowers:does-not-exist")])

    def test_a_real_skill_name_passes(self):
        plan = {"units": [{"id": "U-one", "produces": "code",
                           "skills": ["superpowers:test-driven-development"]}],
                "milestones": [], "claims": [], "combinations": []}
        self.assertEqual(workplan.unknown_skills(plan), [])

    def test_a_unit_naming_no_skill_is_not_a_problem(self):
        """The default covers it; only an explicit wrong name is a defect."""
        plan = {"units": [{"id": "U-one", "produces": "code"}],
                "milestones": [], "claims": [], "combinations": []}
        self.assertEqual(workplan.unknown_skills(plan), [])


if __name__ == "__main__":
    unittest.main()
