#!/usr/bin/env python3
"""Every document listed for a headline must carry it, not merely one of them.

The hole this closes was found on 2026-08-31, hours after `check_headlines.py`
shipped. The tool reported "3 headlines match the run data" while **EVAL.md and
STATUS.md both carried the superseded n=10 figure** and only THESIS_MAP.md and
REPORT.md had been updated to n=22. `quoted_in` returned True as soon as any one
listed document held the value, so a stale document hid behind a fresh one.

That is the same failure the tool was built to catch, one level up: a number
improved somewhere and was not carried to every place that quotes it.

A second hole, found in the same pass: `DOCS["status"]` was declared and then
never referenced by any headline, so STATUS.md — the file whose own header calls
itself the current state of the project — was never read at all.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "tools"))

import check_headlines  # noqa: E402


class TestEveryListedDocumentIsChecked(unittest.TestCase):

    def setUp(self):
        self.tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "hl_cov")
        if not os.path.isdir(self.tmp):
            os.makedirs(self.tmp)

    def write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as handle:
            handle.write(text)
        return path

    def test_one_fresh_document_does_not_excuse_a_stale_one(self):
        """The exact 2026-08-31 situation: fresh MAP, stale EVAL."""
        fresh = self.write("fresh.md", "the oracle beats it on 12 of 22 clips")
        stale = self.write("stale.md", "the oracle beats it on 6 of 10 clips")
        headlines = {
            "claim 1": {"compute": lambda: "12 of 22",
                        "docs": [fresh, stale],
                        "why": "the headline count"},
        }
        result = check_headlines.check(headlines)
        self.assertEqual(len(result["stale"]), 1,
                         "a stale document must fail even beside a fresh one")
        self.assertIn(stale, result["stale"][0][3])
        self.assertNotIn(fresh, result["stale"][0][3])

    def test_all_documents_carrying_it_passes(self):
        a = self.write("a.md", "12 of 22 clips")
        b = self.write("b.md", "also 12 of 22 clips")
        headlines = {"claim 1": {"compute": lambda: "12 of 22",
                                 "docs": [a, b], "why": ""}}
        result = check_headlines.check(headlines)
        self.assertEqual(result["stale"], [])
        self.assertEqual(len(result["ok"]), 1)

    def test_an_absent_document_is_not_silently_passed(self):
        """A path that does not exist must not count as carrying the value."""
        a = self.write("a.md", "12 of 22 clips")
        gone = os.path.join(self.tmp, "does-not-exist.md")
        headlines = {"claim 1": {"compute": lambda: "12 of 22",
                                 "docs": [a, gone], "why": ""}}
        result = check_headlines.check(headlines)
        self.assertEqual(len(result["stale"]), 1)
        self.assertIn(gone, result["stale"][0][3])

    def test_no_run_data_still_skips(self):
        a = self.write("a.md", "nothing here")
        headlines = {"claim 1": {"compute": lambda: None,
                                 "docs": [a], "why": ""}}
        result = check_headlines.check(headlines)
        self.assertEqual(result["skipped"], ["claim 1"])
        self.assertEqual(result["stale"], [])


class TestNoDeclaredDocumentGoesUnused(unittest.TestCase):

    def test_every_entry_in_DOCS_is_referenced_by_some_headline(self):
        """A declared but unreferenced document is never read.

        STATUS.md was declared in DOCS and used by nothing, so the file that
        calls itself the current state of the project was never checked.
        """
        used = set()
        for spec in check_headlines.HEADLINES.values():
            used.update(spec["docs"])
        unused = sorted(set(check_headlines.DOCS.values()) - used)
        self.assertEqual(
            unused, [],
            "declared in DOCS but no headline reads them: %s" % unused)

    def test_status_is_among_the_documents_that_must_carry_claim_1(self):
        docs = check_headlines.HEADLINES["oracle beats the baseline"]["docs"]
        self.assertIn(check_headlines.DOCS["status"], docs)


if __name__ == "__main__":
    unittest.main()
