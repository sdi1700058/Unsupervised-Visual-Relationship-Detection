"""`list_runs` must read the scheduler or say it cannot.

Found 2026-09-06. `list_jobs` did not call `sacct` at all. It globbed
`logs/*.out`, appended a hard-coded row `("25601994", "COMPLETED")`
unconditionally, and printed "DEBUG: Mocked sacct returned N jobs". Its own
docstring claimed it "iterates SLURM sacct output" and was "defensive against
sacct format variation across slurm versions (uses -P parsable mode)", and
`README.md` repeated the claim.

So a tool that reports which training runs exist was inventing one of them, on
every invocation, while documenting itself as authoritative. A fabricated job
id in a run listing is the same class of error as a number in a document that
no file supports, and it is worse for being automated.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from tools import list_runs


class TestNoFabrication(unittest.TestCase):

    def test_no_job_id_is_hard_coded_anywhere_in_the_module(self):
        """The specific row that was being invented."""
        with open(list_runs.__file__, encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("25601994", source)

    def test_the_module_does_not_describe_itself_as_mocked(self):
        with open(list_runs.__file__, encoding="utf-8") as handle:
            source = handle.read().lower()
        self.assertNotIn("mocked sacct", source)

    def test_it_actually_invokes_sacct(self):
        """The docstring's central claim, asserted rather than trusted."""
        import inspect
        source = inspect.getsource(list_runs.list_jobs)
        self.assertIn("sacct", source)


class TestHonestFailure(unittest.TestCase):

    def test_a_missing_scheduler_raises_rather_than_returning_rows(self):
        """No scheduler means no answer, never an invented one.

        `sacct` exists only on the cluster. Locally the tool must say so; the
        one thing it must never do is produce a plausible-looking listing.
        """
        self.assertRaises(list_runs.NoScheduler, list_runs.list_jobs,
                          "2026-01-01", sacct="definitely-not-a-real-command")

    def test_the_error_names_what_is_missing(self):
        try:
            list_runs.list_jobs("2026-01-01", sacct="definitely-not-a-real")
        except list_runs.NoScheduler as exc:
            self.assertIn("sacct", str(exc).lower())
        else:
            self.fail("a missing scheduler returned rows")


class TestParsing(unittest.TestCase):
    """The parsable-mode format the docstring promises to handle."""

    def test_it_reads_pipe_separated_rows(self):
        text = "JobID|State\n25601994|COMPLETED\n25601995|FAILED\n"
        rows = list_runs.parse_sacct(text)
        self.assertEqual(rows, [("25601994", "COMPLETED"),
                                ("25601995", "FAILED")])

    def test_a_job_step_is_not_a_job(self):
        """`.batch` and `.extern` are steps of one job, not three jobs."""
        text = ("JobID|State\n25601994|COMPLETED\n"
                "25601994.batch|COMPLETED\n25601994.extern|COMPLETED\n")
        self.assertEqual(list_runs.parse_sacct(text),
                         [("25601994", "COMPLETED")])

    def test_the_header_is_not_a_row(self):
        self.assertEqual(list_runs.parse_sacct("JobID|State\n"), [])

    def test_empty_output_gives_no_rows(self):
        self.assertEqual(list_runs.parse_sacct(""), [])

    def test_a_state_with_a_reason_keeps_only_the_state(self):
        """sacct writes `CANCELLED by 12345`, and the reason is not a state."""
        text = "JobID|State\n25601994|CANCELLED by 12345\n"
        self.assertEqual(list_runs.parse_sacct(text),
                         [("25601994", "CANCELLED")])


class TestCommandLine(unittest.TestCase):

    def test_a_missing_scheduler_exits_nonzero_with_a_message(self):
        """The refusal has to reach the shell, not only the Python caller."""
        import io
        from contextlib import redirect_stderr
        err = io.StringIO()
        with redirect_stderr(err):
            code = list_runs.main(["--sacct", "definitely-not-a-real-command"])
        self.assertEqual(code, 1)
        self.assertIn("sacct", err.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
