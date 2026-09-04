#!/usr/bin/env python3
"""The documentation gate must catch real rot and stay quiet about history.

A gate that fires on correct history gets switched off within a week, and then
it protects nothing. Each of these cases came out of the 2026-08-31 audit: the
first three are defects it found, the rest are lines it wrongly flagged on the
first run and must never flag again.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "tools"))

import check_docs  # noqa: E402


class DocFixture(unittest.TestCase):

    def setUp(self):
        self.tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "check_docs")
        if not os.path.isdir(self.tmp):
            os.makedirs(self.tmp)

    def doc(self, name, *lines):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        return path


class TestDeadPaths(DocFixture):

    def test_a_moved_script_is_caught(self):
        """STATUS.md pointed at sweeps that had moved to sh/deprecated/."""
        d = self.doc("a.md", "| Round 1 | `sh/overnight_sweep.sh` | 16 |")
        hits = check_docs.dead_paths([d])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["ref"], "sh/overnight_sweep.sh")

    def test_a_path_that_exists_is_quiet(self):
        d = self.doc("b.md", "see `tools/check_docs.py` for the gate")
        self.assertEqual(check_docs.dead_paths([d]), [])

    def test_generated_output_is_not_a_defect(self):
        """eval/ and data/ are gitignored, so absence proves nothing."""
        d = self.doc("c.md",
                     "scored into `eval/planner/E1-structured/summary.csv`",
                     "baked to `data/npz/video/vidvrd/bicycle-30fps.npz`")
        self.assertEqual(check_docs.dead_paths([d]), [])

    def test_a_planned_destination_is_allowed_with_a_reason(self):
        d = self.doc("d.md", "moves to `latplan/domains/video/vidvrd.py` in F1")
        self.assertEqual(check_docs.dead_paths([d]), [])
        self.assertIn("latplan/domains/video/vidvrd.py",
                      check_docs.ALLOWED_ABSENT)

    def test_every_allowance_carries_a_reason(self):
        for path, why in check_docs.ALLOWED_ABSENT.items():
            self.assertTrue(why and len(why) > 20,
                            "%s is allowed with no usable reason" % path)


class TestCompletionClaims(DocFixture):

    def test_declaring_a_pipeline_finished_is_caught(self):
        d = self.doc("e.md", "The evaluation pipeline is finished and proven.")
        self.assertEqual(len(check_docs.completion_claims([d])), 1)

    def test_quoting_the_rule_is_not_breaking_it(self):
        d = self.doc("f.md",
                     "## You do not decide when a task is complete. The user does.",
                     "Never write that a task is complete.",
                     "This heading previously read 'task 3, complete'. Corrected.")
        self.assertEqual(check_docs.completion_claims([d]), [])

    def test_reporting_a_chunk_of_work_is_allowed(self):
        """Finishing an implementation chunk is not closing a work item."""
        d = self.doc("g.md",
                     "Run the gate whenever a chunk of implementation is finished.",
                     "The script ran end to end and produced a number.")
        self.assertEqual(check_docs.completion_claims([d]), [])


class TestStaleCounts(DocFixture):

    def test_a_stale_suite_size_is_caught(self):
        d = self.doc("h.md",
                     "Unit tests: `python3 -m unittest discover -s x` (60 tests,")
        hits = check_docs.stale_counts([d], actual=245)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["claim"], 60)

    def test_a_component_count_is_not_a_suite_claim(self):
        """'20 tests' beside one feature says nothing about the total."""
        d = self.doc("i.md", "| H5 | oracle. Reports the floor. 20 tests. |")
        self.assertEqual(check_docs.stale_counts([d], actual=245), [])

    def test_a_dated_row_is_history_and_must_not_be_rewritten(self):
        d = self.doc("j.md",
                     "| 2026-05-19 to 2026-08-03 | Phase H | 38 tests pass. |")
        self.assertEqual(check_docs.stale_counts([d], actual=245), [])

    def test_a_progression_is_not_a_claim(self):
        d = self.doc("k.md", "Gate passed: 123 -> 162 tests pass, clean.")
        self.assertEqual(check_docs.stale_counts([d], actual=245), [])

    def test_the_correct_count_passes(self):
        d = self.doc("l.md", "Baseline: the test suite is at 245 tests.")
        self.assertEqual(check_docs.stale_counts([d], actual=245), [])

    def test_no_runnable_suite_means_no_opinion(self):
        """A check that cannot measure must be silent, not wrong."""
        d = self.doc("m.md", "the test suite is at 60 tests")
        self.assertEqual(check_docs.stale_counts([d], actual=None), [])


class TestScope(unittest.TestCase):

    def test_append_only_records_are_excluded(self):
        """CHANGES.md names what existed when written. That is not rot."""
        for name in ("CHANGES.md", "AUDIT.md"):
            self.assertIn(name, check_docs.HISTORICAL)

    def test_the_live_documents_are_found(self):
        docs = check_docs.live_docs()
        self.assertTrue(any(d.endswith("STATUS.md") for d in docs))
        self.assertFalse(any(d.endswith("CHANGES.md") for d in docs))


if __name__ == "__main__":
    unittest.main()
