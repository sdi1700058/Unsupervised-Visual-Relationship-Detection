#!/usr/bin/env python3
"""Tests for the Something-Else vocabulary classifier.

The classifier carries a dataset decision, so the cases that decide it are
pinned here. It is a JUDGEMENT rather than a measurement, and these tests
check that the judgement is the one the docstring claims to make -- not that
it is correct about the world.

    python3 -m unittest tools/planner/tests/test_screen_something_else.py
"""

import os
import sys
import unittest
from pathlib import Path

_VIDEO = str(Path(__file__).resolve().parents[2] / "video")
if _VIDEO not in sys.path:
    sys.path.insert(0, _VIDEO)


class TestCategoryTier(unittest.TestCase):

    def test_containment_is_geometric(self):
        """The distinction that makes Something-Else viable at all.

        "Pretending to put something into something" differs from "Putting
        something into something" by whether the object ENDS UP INSIDE the
        other, and containment is a relation between two boxes. The intent is
        inferred by the annotator; the evidence is geometric.
        """
        from screen_something_else import category_tier

        for c in ("Putting something into something",
                  "Pretending to put something into something",
                  "Taking something out of something"):
            self.assertEqual(category_tier(c)[0], "geometric", c)

    def test_orientation_is_semantic_because_a_box_has_none(self):
        """An axis-aligned bounding box does not encode rotation."""
        from screen_something_else import category_tier

        for c in ("Turning something upside down",
                  "Pretending to turn something upside down"):
            self.assertEqual(category_tier(c)[0], "semantic", c)

    def test_material_properties_are_semantic(self):
        from screen_something_else import category_tier

        self.assertEqual(
            category_tier("Trying to bend something unbendable so nothing "
                          "happens")[0], "semantic")
        self.assertEqual(
            category_tier("Pretending to be tearing something that is not "
                          "tearable")[0], "semantic")

    def test_an_unmatched_category_falls_to_semantic_not_geometric(self):
        """The safe direction. Guessing 'geometric' would flatter the dataset."""
        from screen_something_else import category_tier

        tier, marker = category_tier("Zorbling the flimflam")
        self.assertEqual(tier, "semantic")
        self.assertIn("no marker", marker)

    def test_the_first_matching_rule_wins_and_order_is_deliberate(self):
        """`upside down` must beat `turn`, and it is listed before it."""
        from screen_something_else import category_tier, RULES

        markers = [m for _, m in RULES]
        self.assertLess(markers.index("upside down"), markers.index("mov"))


class TestRuleViolation(unittest.TestCase):

    def test_pretending_and_failing_both_count(self):
        from screen_something_else import is_rule_violation

        self.assertTrue(is_rule_violation("Pretending to poke something"))
        self.assertTrue(is_rule_violation(
            "Trying but failing to attach something to something because it "
            "doesn't stick"))

    def test_a_plain_action_does_not_count(self):
        from screen_something_else import is_rule_violation

        self.assertFalse(is_rule_violation("Moving something up"))
        self.assertFalse(is_rule_violation("Uncovering something"))


class TestAgainstTheRealVocabulary(unittest.TestCase):
    """Runs only when the fetched label file is present."""

    LABELS = "data/video/something_else/splits/labels.json"

    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        self.path = root / self.LABELS
        if not self.path.is_file():
            self.skipTest("labels.json not fetched")

    def test_there_are_exactly_174_categories(self):
        import json
        with open(str(self.path)) as f:
            self.assertEqual(len(json.load(f)), 174)

    def test_the_rule_violation_subset_is_mostly_geometric(self):
        """The finding this file exists to support.

        Something-Else's rule-violation categories are what make it score on
        Criterion 0, and unlike VidVRD's coupled predicates they are mostly
        within reach of a positional code. If this ever stops holding, the
        argument for the dataset switch weakens and somebody should be told.
        """
        import json
        from screen_something_else import classify

        with open(str(self.path)) as f:
            rows = classify(list(json.load(f)))
        viol = [r for r in rows if r["rule_violation"]]
        self.assertGreater(len(viol), 20)
        geo = sum(1 for r in viol if r["tier"] == "geometric")
        self.assertGreater(geo / float(len(viol)), 0.5)


if __name__ == "__main__":
    unittest.main()
