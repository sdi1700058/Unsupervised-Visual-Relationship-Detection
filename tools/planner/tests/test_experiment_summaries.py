"""The two experiment summarisers, and the one that reads nothing as a result.

`experiments/G1_heldout_seeds/g1_summary.py` is 454 lines and
`experiments/G4_up_sweep/g4_summary.py` is 804, and neither had a test. Both
READMEs claimed every branch had been exercised on synthetic cells, with no
artefact to show for it. This is that artefact.

**The defect worth pinning.** `eval_plannability.sh` writes a header and exits
0 when no window scores, so a `summary.csv` of zero rows read as
`windows = 0, solve_rate = 0`, and `reading()` reported it as *"the held-out
arm solved no window ... a measured negative result with a named cause"*. An
empty run was being published as a finding. Both halves are now closed -- the
shell exits 4, and the reader says "No reading" -- and this file keeps them
closed.

`slice_export.py` had no test either, and it is the only thing standing
between a per-clip slice and the cross-clip window problem in E1. It silently
truncated an action model in half once already.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    os.pardir)))

import importlib.util

ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    os.pardir))


def _from_file(name, relpath):
    """`experiments/` is not a package, so these load by path."""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, relpath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


g1 = _from_file("_g1_summary", "experiments/G1_heldout_seeds/g1_summary.py")
g4 = _from_file("_g4_summary", "experiments/G4_up_sweep/g4_summary.py")

from tools.planner import slice_export as se  # noqa: E402


class _Npz(dict):
    """The little of `np.load`'s interface that `slice_export` touches."""

    def __init__(self, **arrays):
        import numpy as np
        arrays.setdefault("frame_ids", np.array(["v/%06d" % i
                                                 for i in range(6)]))
        dict.__init__(self, arrays)

    @property
    def files(self):
        return list(self.keys())


def _flatten(value):
    """`reading` returns a string, a list, or a (verdict, caveats) pair."""
    if isinstance(value, str):
        return [value]
    out = []
    for item in value:
        out.extend(_flatten(item))
    return out


def row(export="a.npz", init=0, goal=8, ratio=0.5, method="bfs"):
    return {"export": export, "init": str(init), "goal": str(goal),
            "mse_ratio": str(ratio), "reachability": "True",
            "beats_baseline": "True", "bbox_mse": str(10.0 * ratio),
            "baseline_mse": "10.0", "bbox_iou": "0.5", "baseline_iou": "0.4",
            "method": method, "moving_gt_steps": "8"}


class TestAnEmptyRunIsNotAResult(unittest.TestCase):
    """Zero rows means nothing ran. It does not mean nothing was solved."""

    def test_no_rows_gives_no_windows(self):
        self.assertEqual(g1.summarise_rows([])["windows"], 0)

    def test_the_reading_refuses_rather_than_reporting_zero(self):
        empty = g1.combine([g1.summarise_rows([])])
        text = " ".join(_flatten(g1.reading(empty, empty)))
        self.assertIn("no reading", text.lower())

    def test_a_real_zero_solve_rate_still_gets_its_verdict(self):
        """The guard must not swallow a genuine negative.

        A run that scored windows and solved none is a finding. A run that
        scored no windows is not. The two used to read identically.
        """
        unsolved = dict(row(), reachability="False")
        seen = g1.combine([g1.summarise_rows([row()])])
        test = g1.combine([g1.summarise_rows([unsolved])])
        self.assertEqual(test["windows"], 1)
        text = " ".join(_flatten(g1.reading(seen, test)))
        self.assertNotIn("no reading", text.lower())


class TestWindowsAreCountedOnce(unittest.TestCase):
    """A row is one window scored by one planner, and G1 runs two."""

    def test_two_planners_on_one_window_is_one_window(self):
        a, b = row(method="bfs"), row(method="pddl", ratio=0.9)
        self.assertEqual(g1.summarise_rows([a, b])["windows"], 1)

    def test_the_better_planner_is_the_one_credited(self):
        a, b = row(method="bfs", ratio=0.5), row(method="pddl", ratio=0.9)
        got = g1.summarise_rows([a, b])
        self.assertAlmostEqual(got["ratio"], 0.5, places=6)

    def test_two_clips_at_the_same_window_index_are_two_windows(self):
        """Per-clip slicing depends on this, and it is why E1 is stuck."""
        rows = [row(export="clipA.npz"), row(export="clipB.npz")]
        self.assertEqual(g1.summarise_rows(rows)["windows"], 2)


class TestLivenessIsNotSilentlySkipped(unittest.TestCase):
    """G4 dropped its whole dead-latent section when numpy was absent.

    A dead cell then entered the table as an ordinary planning result. The
    reading now says so out loud instead.
    """

    def test_a_cell_with_no_distinct_count_says_liveness_was_not_checked(self):
        cells = [{"u": 40, "p": 10, "bits": 400, "distinct": None,
                  "windows": 4, "solved": 2, "ratio": 0.8, "val": 0.12}]
        self.assertIn("liveness", " ".join(_flatten(g4.reading(cells))).lower())

    def test_a_live_cell_does_not_raise_that_warning(self):
        cells = [{"u": 40, "p": 10, "bits": 400, "distinct": 96,
                  "windows": 4, "solved": 2, "ratio": 0.8, "val": 0.12}]
        self.assertNotIn("liveness was not checked",
                         " ".join(_flatten(g4.reading(cells))).lower())

    def test_no_cells_at_all_does_not_crash(self):
        self.assertTrue(_flatten(g4.reading([])))


class TestSliceExport(unittest.TestCase):
    """Its only caller is G1, and it once truncated an action model in half."""

    def test_a_clip_is_found_from_its_frame_id(self):
        self.assertEqual(se.clip_of("ILSVRC2015_train_00005005/000012"),
                         "ILSVRC2015_train_00005005")

    def test_the_index_gives_one_span_per_clip(self):
        """An ordered list of (clip, (start, stop)), half-open."""
        ids = ["a/000000", "a/000001", "b/000000", "b/000001", "b/000002"]
        self.assertEqual(se.clip_index(ids),
                         [("a", (0, 2)), ("b", (2, 5))])

    def test_a_clip_split_in_two_places_is_refused_not_merged(self):
        """Interleaved frames would make one span cover another clip."""
        ids = ["a/000000", "b/000000", "a/000001"]
        self.assertRaises(Exception, se.clip_index, ids)

    def test_a_per_frame_array_is_sliced(self):
        import numpy as np
        data = _Npz(latents=np.arange(12).reshape(6, 2),
                    gt_boxes=np.zeros((6, 1, 4)))
        out = se.slice_export(data, 2, 5)
        self.assertEqual(out["latents"].shape[0], 3)
        self.assertEqual(out["gt_boxes"].shape[0], 3)

    def test_the_action_model_is_not_sliced_with_the_frames(self):
        """The bug: `actions` had one row per transition, not per frame.

        With six frames and six action rows it was cut to three, so half the
        mined action model vanished with no error anywhere.
        """
        import numpy as np
        data = _Npz(latents=np.arange(12).reshape(6, 2),
                    actions=np.arange(24).reshape(6, 4))
        out = se.slice_export(data, 0, 3)
        self.assertEqual(out["latents"].shape[0], 3)
        self.assertEqual(out["actions"].shape[0], 6)

    def test_a_scalar_is_copied_rather_than_sliced(self):
        import numpy as np
        data = _Npz(latents=np.zeros((6, 2)), n_bits=np.int64(400))
        self.assertEqual(int(se.slice_export(data, 1, 3)["n_bits"]), 400)


if __name__ == "__main__":
    unittest.main()
