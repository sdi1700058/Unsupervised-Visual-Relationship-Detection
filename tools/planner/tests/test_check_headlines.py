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
import shutil
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


class TestContradiction(unittest.TestCase):
    """Presence is not enough: a wrong value beside the right one must fail.

    Found 2026-09-06. `missing_from` asked only whether the correct value
    appeared somewhere in the document. Corrupting the headline table while
    the generated claim block still carried the right number passed the check,
    and the claim blocks are present in three of the four tracked documents,
    so the check was neutralised where it mattered most.
    """

    def setUp(self):
        from tools import check_headlines
        self.mod = check_headlines
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, text):
        path = os.path.join(self.dir, "doc.md")
        with open(path, "w") as handle:
            handle.write(text)
        return path

    def test_a_wrong_count_beside_the_right_one_is_reported(self):
        path = self.write("the oracle beat it on 12 of 22 clips\n"
                          "| beats the straight line | 12 of 99 |\n")
        hits = self.mod.contradicted_in([path], "12 of 22",
                                        anchor="beats the straight line")
        self.assertEqual(len(hits), 1)
        self.assertIn("12 of 99", hits[0]["found"])

    def test_the_right_count_alone_is_not_reported(self):
        path = self.write("the oracle beat it on 12 of 22 clips\n")
        self.assertEqual(self.mod.contradicted_in(
            [path], "12 of 22", anchor="beats the straight line"), [])

    def test_a_count_on_an_unanchored_line_is_left_alone(self):
        """A different sample elsewhere in the document is not a
        contradiction. Only the line claiming this headline is judged."""
        path = self.write("a subset gave 6 of 10\n"
                          "beats the straight line | 12 of 22\n")
        self.assertEqual(self.mod.contradicted_in(
            [path], "12 of 22", anchor="beats the straight line"), [])

    def test_a_headline_that_is_not_a_count_is_skipped(self):
        """A bare decimal cannot be told from any other decimal, so the
        contradiction test does not apply to it and says so by doing nothing."""
        path = self.write("median 0.84 here and 0.99 elsewhere\n")
        self.assertEqual(self.mod.contradicted_in(
            [path], "0.84", anchor="median"), [])

    def test_a_missing_file_is_not_a_contradiction(self):
        self.assertEqual(
            self.mod.contradicted_in(
                [os.path.join(self.dir, "absent.md")], "12 of 22",
                anchor="beats"), [])

    def test_other_columns_on_the_same_row_are_not_contradictions(self):
        """One row carries several headlines. If the value is present the row
        states this one correctly, and the neighbouring counts are other
        columns rather than errors."""
        path = self.write(
            "| beats the straight line | 12 of 22 | 0 of 10 | 0 of 10 |\n")
        self.assertEqual(self.mod.contradicted_in(
            [path], "12 of 22", anchor="beats the straight line"), [])

    def test_a_row_missing_the_value_entirely_is_reported(self):
        path = self.write(
            "| beats the straight line | 12 of 99 | 0 of 10 |\n")
        hits = self.mod.contradicted_in([path], "12 of 22",
                                        anchor="beats the straight line")
        self.assertTrue(hits)

    def test_check_reports_a_contradiction_as_stale(self):
        """`check` is what the gate calls, and the wiring was untested.

        Every test above calls `contradicted_in` directly, and the two
        `check` tests elsewhere in the suite pass headlines with no anchor,
        so `contradicted_in` returned nothing for them. Changing `check` to
        ignore its result left the whole suite green.
        """
        path = self.write("the oracle beat it on 12 of 22 clips\n"
                          "| beats the straight line | 12 of 99 |\n")
        headlines = {"claim 1": {"compute": lambda: "12 of 22",
                                 "docs": [path],
                                 "anchor": "beats the straight line",
                                 "why": ""}}
        result = self.mod.check(headlines)
        self.assertEqual(len(result["stale"]), 1)
        # The document carries the right value, so nothing is missing. It is
        # the wrong value beside it that makes the headline stale.
        self.assertEqual(result["stale"][0][3], [])
        self.assertTrue(result["stale"][0][4])

    def test_check_passes_a_document_that_only_carries_the_right_value(self):
        path = self.write("| beats the straight line | 12 of 22 |\n")
        headlines = {"claim 1": {"compute": lambda: "12 of 22",
                                 "docs": [path],
                                 "anchor": "beats the straight line",
                                 "why": ""}}
        result = self.mod.check(headlines)
        self.assertEqual(result["stale"], [])
        self.assertEqual(len(result["ok"]), 1)
