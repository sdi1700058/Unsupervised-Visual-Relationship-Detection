"""The curve plot draws the trivial model's loss when a run recorded it.

`strips.py vidvrd()` writes `baseline.json` beside `training_history.csv`
from 2026-10-03, the loss of a model that always predicts the mean training
state. `tools/plot_training_curve.py` reads it and draws it as horizontal
lines, so the author can see by eye whether a curve fell below it. A run from
before that day has no such file and must still plot.
"""

import importlib.util
import json
import os
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
_spec = importlib.util.spec_from_file_location(
    "_plot_training_curve", os.path.join(ROOT, "tools", "plot_training_curve.py"))
ptc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptc)


class TestLoadBaseline(unittest.TestCase):

    def test_a_run_with_baseline_json_gives_its_two_bce_values(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "baseline.json"), "w") as f:
                json.dump({"train_BCE": 0.6, "val_BCE": 0.65,
                           "train_MSE": 0.2, "val_MSE": 0.21}, f)
            self.assertEqual(ptc.load_baseline(d),
                             {"train_BCE": 0.6, "val_BCE": 0.65,
                              "train_MSE": 0.2, "val_MSE": 0.21})

    def test_a_run_without_it_gives_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ptc.load_baseline(d))

    def test_a_log_file_target_gives_none(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "run.out")
            open(log, "w").close()
            self.assertIsNone(ptc.load_baseline(log))


if __name__ == "__main__":
    unittest.main()
