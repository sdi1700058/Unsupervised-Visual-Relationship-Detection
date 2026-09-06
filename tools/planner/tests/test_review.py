"""The review ledger, which has to survive the session dying.

A review of 230 artefacts cannot run in one pass. It has to be resumable, so
the state lives on disk and every artefact is recorded before the next is
started. If the process dies halfway, the next run continues rather than
restarting.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import review


class TestBuild(unittest.TestCase):

    def test_every_file_becomes_an_entry(self):
        led = review.build([("a.py", "code"), ("b.md", "documents")])
        self.assertEqual(len(led["entries"]), 2)

    def test_a_new_entry_is_unreviewed(self):
        led = review.build([("a.py", "code")])
        self.assertEqual(led["entries"][0]["status"], "unreviewed")
        self.assertEqual(led["entries"][0]["findings"], [])

    def test_the_pass_is_recorded_on_each_entry(self):
        led = review.build([("a.py", "code"), ("b.md", "documents")])
        by_path = dict((e["path"], e["pass"]) for e in led["entries"])
        self.assertEqual(by_path["a.py"], "code")
        self.assertEqual(by_path["b.md"], "documents")

    def test_rebuilding_keeps_what_was_already_reviewed(self):
        """The whole point. A rebuild after new files appear must not throw
        away the work already done."""
        led = review.build([("a.py", "code")])
        review.record(led, "a.py", [], "sound")
        led = review.build([("a.py", "code"), ("new.py", "code")], led)
        done = [e for e in led["entries"] if e["status"] == "reviewed"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["path"], "a.py")
        self.assertEqual(len(led["entries"]), 2)


class TestNextBatch(unittest.TestCase):

    def setUp(self):
        self.led = review.build([("a.py", "code"), ("b.py", "code"),
                                 ("c.md", "documents")])

    def test_it_returns_only_unreviewed_entries(self):
        review.record(self.led, "a.py", [], "sound")
        paths = [e["path"] for e in review.next_batch(self.led, 5)]
        self.assertNotIn("a.py", paths)

    def test_it_honours_the_batch_size(self):
        self.assertEqual(len(review.next_batch(self.led, 2)), 2)

    def test_it_can_be_limited_to_one_pass(self):
        got = review.next_batch(self.led, 5, which="documents")
        self.assertEqual([e["path"] for e in got], ["c.md"])

    def test_an_unknown_pass_returns_nothing_rather_than_raising(self):
        self.assertEqual(review.next_batch(self.led, 5, which="banana"), [])

    def test_an_exhausted_ledger_returns_nothing(self):
        for path in ("a.py", "b.py", "c.md"):
            review.record(self.led, path, [], "sound")
        self.assertEqual(review.next_batch(self.led, 5), [])


class TestRecord(unittest.TestCase):

    def setUp(self):
        self.led = review.build([("a.py", "code")])

    def test_recording_marks_the_entry_reviewed(self):
        review.record(self.led, "a.py", [], "sound")
        self.assertEqual(self.led["entries"][0]["status"], "reviewed")

    def test_findings_are_kept_with_their_severity(self):
        review.record(self.led, "a.py",
                      [{"severity": "high", "what": "imports a moved file"}],
                      "defect")
        found = self.led["entries"][0]["findings"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["severity"], "high")

    def test_an_unknown_severity_is_refused(self):
        """A severity nobody defined cannot be sorted or counted."""
        self.assertRaises(
            ValueError, review.record, self.led, "a.py",
            [{"severity": "catastrophic", "what": "x"}], "defect")

    def test_an_unknown_path_is_refused(self):
        self.assertRaises(ValueError, review.record, self.led, "nope.py",
                          [], "sound")

    def test_a_verdict_is_required_to_count_as_reviewed(self):
        self.assertRaises(ValueError, review.record, self.led, "a.py", [], "")


class TestProgress(unittest.TestCase):

    def test_it_counts_per_pass(self):
        led = review.build([("a.py", "code"), ("b.py", "code"),
                            ("c.md", "documents")])
        review.record(led, "a.py", [], "sound")
        got = review.progress(led)
        self.assertEqual(got["code"]["reviewed"], 1)
        self.assertEqual(got["code"]["total"], 2)
        self.assertEqual(got["documents"]["reviewed"], 0)

    def test_it_counts_findings_by_severity(self):
        led = review.build([("a.py", "code")])
        review.record(led, "a.py",
                      [{"severity": "high", "what": "x"},
                       {"severity": "low", "what": "y"}], "defect")
        got = review.progress(led)
        self.assertEqual(got["_findings"]["high"], 1)
        self.assertEqual(got["_findings"]["low"], 1)


class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "ledger.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_a_saved_ledger_reloads_with_its_findings(self):
        """If the session dies, the next run must continue rather than
        restart. This is the test that matters."""
        led = review.build([("a.py", "code"), ("b.py", "code")])
        review.record(led, "a.py",
                      [{"severity": "medium", "what": "no test"}], "defect")
        review.save(led, self.path)

        again = review.load(self.path)
        self.assertEqual(len(again["entries"]), 2)
        done = [e for e in again["entries"] if e["status"] == "reviewed"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["findings"][0]["what"], "no test")
        self.assertEqual([e["path"] for e in review.next_batch(again, 5)],
                         ["b.py"])

    def test_loading_a_missing_ledger_gives_an_empty_one(self):
        led = review.load(os.path.join(self.dir, "absent.json"))
        self.assertEqual(led["entries"], [])


if __name__ == "__main__":
    unittest.main()
