#!/usr/bin/env python3
"""Tests for the superseded-number checker.

A number that has been withdrawn must not sit in a document reading as
current. Yesterday produced four wrong headlines; nine occurrences of them
survived across the reference documents after each was "corrected where it was
found", and two of those only surfaced from a scripted sweep:

  * a duplicate annotation, two wordings one line apart;
  * a restatement three sections away from the section that was marked.

The sweep was written as a throwaway. This is it kept.

    python3 -m unittest tools/planner/tests/test_check_superseded.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class TestFindUncovered(unittest.TestCase):

    def _doc(self, text):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "doc.md")
        with open(p, "w") as f:
            f.write(text)
        return p

    def test_a_bare_superseded_number_is_reported(self):
        from tools.check_superseded import find_uncovered

        p = self._doc("# Results\n\nThe ratio was 0.046 and it is fine.\n")
        hits = find_uncovered([p], {"0.046": "withdrawn"})
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["value"], "0.046")

    def test_a_marked_section_is_not_reported(self):
        from tools.check_superseded import find_uncovered

        p = self._doc("# Results\n\n> SUPERSEDED: do not quote.\n\n"
                      "The ratio was 0.046.\n")
        self.assertEqual(find_uncovered([p], {"0.046": "withdrawn"}), [])

    def test_the_marker_must_be_in_the_SAME_section(self):
        """The restatement-three-sections-away case, which reading missed."""
        from tools.check_superseded import find_uncovered

        p = self._doc("# One\n\n> SUPERSEDED\n\nratio 0.046 here.\n\n"
                      "# Two\n\nand again 0.046 here.\n")
        hits = find_uncovered([p], {"0.046": "withdrawn"})
        self.assertEqual(len(hits), 1)
        self.assertIn("Two", hits[0]["heading"])

    def test_a_marker_just_after_the_line_also_counts(self):
        """A table row annotated on the following line is covered."""
        from tools.check_superseded import find_uncovered

        p = self._doc("# T\n\n| ratio | 0.046 |\n"
                      "| | *pre-review, superseded* |\n")
        self.assertEqual(find_uncovered([p], {"0.046": "withdrawn"}), [])

    def test_word_boundaries_are_respected(self):
        """`9.91` must not match inside `19.912`."""
        from tools.check_superseded import find_uncovered

        p = self._doc("# T\n\nthe value 19.912 is unrelated.\n")
        self.assertEqual(find_uncovered([p], {"9.91": "withdrawn"}), [])

    def test_the_registry_carries_a_reason(self):
        """So a hit tells the reader why, not just that."""
        from tools.check_superseded import SUPERSEDED

        self.assertTrue(SUPERSEDED)
        for value, reason in SUPERSEDED.items():
            self.assertIsInstance(value, str)
            self.assertGreater(len(reason), 20, value)


if __name__ == "__main__":
    unittest.main()


class TestScope(unittest.TestCase):
    """A checker only guards what it reads.

    `DEFAULT_PATHS` was `notes/REPORT.md` and `notes/docs/*.md`, so the working
    documents at the top of `notes/` were never scanned. The withdrawn figure
    1.377 sat in `NOW.md` for five days, unmarked, while this check reported
    that no withdrawn number reads as current. A number is withdrawn wherever
    it appears, not only in the two places the tuple happened to name.
    """

    def test_the_working_documents_are_in_scope(self):
        from tools import check_superseded
        for name in ("notes/NOW.md", "notes/PROGRESS.md", "notes/QUALITY.md",
                     "notes/WORKBOARD.md", "notes/INTERVIEW.md"):
            self.assertIn(name, check_superseded.DEFAULT_PATHS,
                          "%s is not scanned, so a withdrawn number there "
                          "would go unreported" % name)

    def test_the_documents_directory_is_still_in_scope(self):
        from tools import check_superseded
        self.assertIn("notes/docs/*.md", check_superseded.DEFAULT_PATHS)

    def test_the_append_only_records_stay_exempt(self):
        """A dated entry quoting what was true that day is history."""
        from tools import check_superseded
        self.assertIn("CHANGES.md", check_superseded.HISTORICAL)
        self.assertIn("AUDIT.md", check_superseded.HISTORICAL)
