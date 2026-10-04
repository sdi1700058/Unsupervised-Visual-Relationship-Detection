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


class TestLatentFrom(unittest.TestCase):
    """The epoch from which every reconstruction goes through the latent.

    latplan/model.py build_decoder skips the latent through the pre-encoder
    until epoch * preencoder_delay and fades it in by twice that, so a curve
    before that epoch does not measure FOSAE. Found 2026-10-04 from a jump at
    epoch 600 in all eight rung runs.
    """

    def test_the_mark_is_twice_the_delay_in_epochs(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "aux.json"), "w") as f:
                json.dump({"parameters": {"epoch": 3000,
                                          "preencoder_delay": 0.1,
                                          "preencoder_layers": 2}}, f)
            self.assertEqual(ptc.latent_from(d), 600)

    def test_no_pre_encoder_or_no_file_gives_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(ptc.latent_from(d))


if __name__ == "__main__":
    unittest.main()
