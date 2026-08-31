#!/usr/bin/env python3
"""Tests for the headline-number checker.

`check_superseded.py` catches a number that was WITHDRAWN. It cannot catch a
number that was merely improved on: Claim 1 sat at n=10 (6 of 10) in
`THESIS_MAP.md` for a day while n=22 (12 of 22) was already on disk, and that
was found by hand rather than by a check.

This closes that class. Each headline is recomputed from the run data and the
documents are checked for the current value.

    python3 -m unittest tools/planner/tests/test_check_headlines.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class TestQuotedIn(unittest.TestCase):

    def _doc(self, text):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "doc.md")
        with open(p, "w") as f:
            f.write(text)
        return p

    def test_a_document_quoting_the_value_passes(self):
        from tools.check_headlines import quoted_in

        p = self._doc("The oracle beats it on 12 of 22 clips.\n")
        self.assertTrue(quoted_in([p], "12 of 22"))

    def test_a_document_missing_it_fails(self):
        from tools.check_headlines import quoted_in

        p = self._doc("The oracle beats it on 6 of 10 clips.\n")
        self.assertFalse(quoted_in([p], "12 of 22"))

    def test_any_one_document_is_enough(self):
        from tools.check_headlines import quoted_in

        a = self._doc("nothing here\n")
        b = self._doc("12 of 22\n")
        self.assertTrue(quoted_in([a, b], "12 of 22"))

    def test_a_missing_file_is_not_an_error(self):
        from tools.check_headlines import quoted_in

        self.assertFalse(quoted_in(["/nonexistent/x.md"], "12 of 22"))


class TestFormatting(unittest.TestCase):

    def test_a_ratio_is_rendered_to_two_decimals(self):
        from tools.check_headlines import fmt_ratio

        self.assertEqual(fmt_ratio(0.8412), "0.84")
        self.assertEqual(fmt_ratio(1.5), "1.50")

    def test_a_count_pair_reads_as_n_of_m(self):
        from tools.check_headlines import fmt_count

        self.assertEqual(fmt_count(12, 22), "12 of 22")


class TestRegistry(unittest.TestCase):

    def test_every_headline_names_the_documents_it_must_appear_in(self):
        from tools.check_headlines import HEADLINES

        self.assertTrue(HEADLINES)
        for name, spec in HEADLINES.items():
            self.assertIn("docs", spec, name)
            self.assertIn("compute", spec, name)
            self.assertTrue(callable(spec["compute"]), name)

    def test_a_headline_with_no_run_data_is_skipped_not_failed(self):
        """Deleting an eval directory must not fail the gate."""
        from tools.check_headlines import check

        fake = {"nothing": {"compute": lambda: None,
                            "docs": ["/nonexistent.md"],
                            "why": "test"}}
        result = check(fake)
        self.assertEqual(result["skipped"], ["nothing"])
        self.assertEqual(result["stale"], [])


if __name__ == "__main__":
    unittest.main()
