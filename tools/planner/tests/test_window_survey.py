#!/usr/bin/env python3
"""The crossover criterion must have ONE definition, not two.

`EVAL.md` 4.2's criterion is computed in two places: `screen_vidvrd.
window_crossover`, which selects clips, and `window_survey.survey_clip`, which
sweeps window sizes across the corpus. The 2026-08-30 review found they
disagreed by construction -- the survey ran with `fill=True`, so it measured
the criterion on frames the loader had invented, while the screen was
absence-aware.

A criterion the thesis quotes must not depend on which tool computed it.

    python3 -m unittest tools/planner/tests/test_window_survey.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
_VIDEO = str(Path(__file__).resolve().parents[2] / "video")
if _VIDEO not in sys.path:
    sys.path.insert(0, _VIDEO)


def _has_pillow():
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


def _clip(path, n=40, gap=None):
    """Two objects on a smooth diagonal. `gap` removes object 1 for a run."""
    traj = []
    for f in range(n):
        ents = [{"tid": 0, "bbox": {"xmin": 10.0 + 4 * f, "ymin": 20.0,
                                    "xmax": 40.0 + 4 * f, "ymax": 50.0}}]
        if gap is None or not (gap[0] <= f < gap[1]):
            ents.append({"tid": 1, "bbox": {"xmin": 200.0 - 3 * f, "ymin": 90.0,
                                            "xmax": 230.0 - 3 * f, "ymax": 120.0}})
        traj.append(ents)
    doc = {"video_id": "SURVEY_0001", "width": 640, "height": 360, "fps": 30,
           "subject/objects": [{"tid": 0, "category": "dog"},
                               {"tid": 1, "category": "frisbee"}],
           "relation_instances": [], "trajectories": traj}
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


@unittest.skipUnless(_has_pillow(), "needs pillow (present in .venv-local)")
class TestSurveyDoesNotFabricate(unittest.TestCase):

    def test_fill_is_off_by_default(self):
        """The defect: the survey used to invent the frames it measured."""
        import inspect
        from tools.planner.window_survey import survey_clip

        sig = inspect.signature(survey_clip)
        self.assertIn("fill", sig.parameters)
        self.assertIs(sig.parameters["fill"].default, False)

    def test_a_gap_changes_the_answer_when_filled(self):
        """Filling a gap is not free, and the survey must not do it silently."""
        from tools.planner.window_survey import survey_clip

        with tempfile.TemporaryDirectory() as tmp:
            p = _clip(os.path.join(tmp, "SURVEY_0001.json"), n=40, gap=(10, 20))
            honest = survey_clip(p, 2, 60, 40, fill=False)
            filled = survey_clip(p, 2, 60, 40, fill=True)

        self.assertIsNotNone(honest)
        self.assertIsNotNone(filled)
        # Filling manufactures ten motionless frames for object 1, which drags
        # the baseline down and the ratio up.
        self.assertNotEqual(round(honest["floor"], 6),
                            round(filled["floor"], 6))

    def test_absent_objects_do_not_enter_the_baseline(self):
        """A zero box is not a box at the origin (SPEC V30)."""
        from tools.planner.window_survey import survey_clip

        with tempfile.TemporaryDirectory() as tmp:
            a = survey_clip(_clip(os.path.join(tmp, "a.json"), n=40),
                            2, 60, 40, fill=False)
            b = survey_clip(_clip(os.path.join(tmp, "b.json"), n=40,
                                  gap=(12, 18)), 2, 60, 40, fill=False)

        # Object 1 vanishing for six frames must not look like a 200 px jump to
        # the origin. Before the fix the floor/baseline ratio moved by orders
        # of magnitude; absence-aware it barely moves.
        self.assertLess(abs(a["floor"] - b["floor"]), a["floor"])


@unittest.skipUnless(_has_pillow(), "needs pillow (present in .venv-local)")
class TestOneCriterionTwoTools(unittest.TestCase):
    """The cross-check the review asked for."""

    def _per_object(self, doc):
        per = {}
        for f, ents in enumerate(doc["trajectories"]):
            for o in ents:
                b = o["bbox"]
                per.setdefault(o["tid"], {})[f] = [
                    b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
        return per

    def test_the_two_implementations_agree_on_winnability(self):
        from screen_vidvrd import window_crossover
        from tools.planner.window_survey import survey_clip

        with tempfile.TemporaryDirectory() as tmp:
            p = _clip(os.path.join(tmp, "SURVEY_0001.json"), n=40)
            with open(p) as f:
                doc = json.load(f)
            screen = window_crossover(self._per_object(doc),
                                      doc["width"], doc["height"], window=8)
            survey = survey_clip(p, 2, 60, 40, fill=False, windows=(8,))

        self.assertIsNotNone(screen)
        self.assertIsNotNone(survey)
        ratio = survey["per_window"].get(8)
        self.assertIsNotNone(ratio)

        # Same criterion, different tools and coordinate spaces. They must at
        # minimum return the SAME VERDICT; the exact ratios differ because the
        # screen works in the video's own pixels and the survey on the canvas.
        self.assertEqual(screen < 1.0, ratio < 1.0,
                         "screen says winnable=%s, survey says winnable=%s "
                         "(screen %.4f, survey %.4f)"
                         % (screen < 1.0, ratio < 1.0, screen, ratio))


if __name__ == "__main__":
    unittest.main()
