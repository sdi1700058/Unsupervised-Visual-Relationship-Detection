"""An empty plan is valid, and it is not a planned window.

`bfs.search` carries this guard and records what it cost: **108 of 385
recorded rows were `reachability=True` with `plan_length=0`**, and one run
reported "6 of 12 solved" where the true count of planned windows was zero.

When the two endpoint frames encode the same latent, the empty plan is
genuinely the shortest plan, so returning success is arithmetically correct
and reporting it as a solved window is not. Neither `pddl/planner.py` nor
`ama3/planner.py` had the guard, so both were still producing that row.

The guard sits first in `_solve`, before the domain is written and before any
external planner is called, which is what lets this run with no Fast Downward
and no roswell on the machine.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    os.pardir)))

from tools.planner.pddl import planner as pddl_planner
from tools.planner.ama3 import planner as ama3_planner


class _ExportThatMustNotBeUsed(object):
    """If the guard works, nothing here is ever called."""

    def transitions(self):
        raise AssertionError(
            "the guard let execution past it: identical endpoints reached "
            "the planner")


class TestBothPlannersRefuseToCallItSolved(unittest.TestCase):

    def setUp(self):
        self.z = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
        self.z_all = np.stack([self.z, 1 - self.z])

    def _call(self, module, z_init, z_goal):
        return module._solve(z_init, z_goal, self.z_all, 10.0, "/tmp",
                             export=_ExportThatMustNotBeUsed())

    def test_pddl_returns_before_writing_a_domain(self):
        ok, trace, wall, extra = self._call(pddl_planner, self.z, self.z.copy())
        self.assertTrue(ok)
        self.assertEqual(extra["outcome"], "endpoints_identical")

    def test_ama3_returns_before_writing_a_domain(self):
        ok, trace, wall, extra = self._call(ama3_planner, self.z, self.z.copy())
        self.assertTrue(ok)
        self.assertEqual(extra["outcome"], "endpoints_identical")

    def test_the_trace_is_the_single_state(self):
        """The empty plan visits one state, so plan_length is zero."""
        for module in (pddl_planner, ama3_planner):
            _, trace, _, _ = self._call(module, self.z, self.z.copy())
            self.assertEqual(len(trace), 1)

    def test_a_genuine_pair_is_not_intercepted(self):
        """The guard must not swallow the ordinary case.

        Different endpoints must reach the planner. The stub export raises
        when touched, so reaching it proves the guard let go.
        """
        for module in (pddl_planner, ama3_planner):
            self.assertRaises(AssertionError, self._call,
                              module, self.z, 1 - self.z)

    def test_the_outcome_word_is_the_one_bfs_already_uses(self):
        """One word for one thing, across all three planners."""
        from tools.planner.bfs.planner import OUTCOMES
        self.assertIn("endpoints_identical", OUTCOMES)
        for module in (pddl_planner, ama3_planner):
            _, _, _, extra = self._call(module, self.z, self.z.copy())
            self.assertIn(extra["outcome"], OUTCOMES)


if __name__ == "__main__":
    unittest.main()
