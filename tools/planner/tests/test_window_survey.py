#!/usr/bin/env python3
"""The crossover criterion must have ONE definition, not two.

`EVAL.md` 4.2's criterion is computed in two places: `screen_vidvrd.
window_crossover`, which selects clips, and `window_survey.survey_clip`, which
sweeps window sizes across the dataset. The 2026-08-30 review found they
disagreed by construction -- the survey ran with `fill=True`, so it measured
the criterion on frames the loader had invented, while the screen was
absence-aware.

A criterion the thesis quotes must not depend on which tool computed it.

The 2026-09-07 review found the second half of the same defect: the survey
divided a **clip-wide** quantisation floor by a **per-window** baseline, so the
two sides of the ratio were never measured on the same frames or the same
objects. `oracle.round_trip_error`'s docstring names that exact mistake.
`TestTheFloorMatchesTheBaselinesFrames` pins the correction.

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

    def test_an_unannotated_frame_changes_the_answer_when_filled(self):
        """Filling a gap is not free, and the survey must not do it silently."""
        import json

        from tools.planner.window_survey import survey_clip

        with tempfile.TemporaryDirectory() as tmp:
            p = _clip(os.path.join(tmp, "SURVEY_0001.json"), n=40)
            with open(p) as handle:
                clip = json.load(handle)
            for i in range(10, 20):
                clip["trajectories"][i] = []       # unannotated, not partial
            with open(p, "w") as handle:
                json.dump(clip, handle)
            honest = survey_clip(p, 2, 60, 40, fill=False)
            filled = survey_clip(p, 2, 60, 40, fill=True)

        self.assertIsNotNone(honest)
        self.assertIsNotNone(filled)
        # Filling manufactures ten motionless frames, which lowers the floor and
        # raises the fraction of duplicate frames.
        self.assertNotEqual(round(honest["clip_floor"], 6),
                            round(filled["clip_floor"], 6))
        self.assertGreater(filled["duplicate_frac"], honest["duplicate_frac"])

    def test_a_partial_frame_is_not_filled(self):
        """`fill` works on unannotated FRAMES, never on a missing object.

        The earlier version of the test above removed one object from a range
        of frames and expected filling to change the answer. It does not, and
        should not: every frame there still carries a box, so the dataset never
        left it unannotated. A slot absent from an annotated frame means the
        object is not in the shot, and inventing a position for it would be the
        fabrication this whole check exists to prevent.
        """
        from tools.planner.window_survey import survey_clip

        with tempfile.TemporaryDirectory() as tmp:
            p = _clip(os.path.join(tmp, "PARTIAL_0001.json"), n=40,
                      gap=(10, 20))
            honest = survey_clip(p, 2, 60, 40, fill=False)
            filled = survey_clip(p, 2, 60, 40, fill=True)

        self.assertEqual(round(honest["clip_floor"], 6),
                         round(filled["clip_floor"], 6))

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
        self.assertLess(abs(a["clip_floor"] - b["clip_floor"]), a["clip_floor"])


@unittest.skipUnless(_has_pillow(), "needs pillow (present in .venv-local)")
class TestTheFloorMatchesTheBaselinesFrames(unittest.TestCase):
    """`floor / mse` must have one frame set and one object set on both sides.

    Until 2026-09-07 `survey_clip` computed the floor once over the whole clip
    and every object slot, then divided it by a baseline measured on one
    window's *intermediate* frames and only the objects present throughout that
    window. Two different populations, one ratio.

    The clip below separates them by construction, and the separation is the
    ordinary case rather than a contrivance: two of the three objects blink out
    for a single frame, so neither can be scored in the window -- yet under the
    old code both were still paying into the floor that the window's verdict
    was read from. All coordinates are chosen against the decoder's 5 px bins,
    so the two floors are exact rather than approximate.
    """

    WINDOW = 8

    def _split_floors(self, path):
        """Build the clip whose two floors disagree by a factor of 20."""
        traj = []
        for f in range(self.WINDOW + 1):
            # Slot 0: present throughout, so it alone is scored. Its endpoints
            # sit 2 px into a bin and its interior frames 1 px into one, so the
            # window's own floor is small and known.
            x1 = 12.0 if f == 0 else (92.0 if f == self.WINDOW
                                      else 16.0 + 10.0 * f)
            ents = [{"tid": 0, "bbox": {"xmin": x1, "ymin": 100.0,
                                        "xmax": x1 + 30.0, "ymax": 130.0}}]
            # Slots 1 and 2: static, badly quantised (4 px into every bin, the
            # worst a 5 px bin allows), and each missing one interior frame.
            # Missing one frame of the window disqualifies them from the
            # baseline, so they must not reach the floor either.
            if f != 3:
                ents.append({"tid": 1, "bbox": {"xmin": 4.0, "ymin": 4.0,
                                                "xmax": 54.0, "ymax": 54.0}})
            if f != 5:
                ents.append({"tid": 2, "bbox": {"xmin": 104.0, "ymin": 104.0,
                                                "xmax": 154.0, "ymax": 154.0}})
            traj.append(ents)
        # 300x200 source == the canvas, so these coordinates survive scaling.
        doc = {"video_id": "FLOORS_0001", "width": 300, "height": 200,
               "fps": 30,
               "subject/objects": [{"tid": 0, "category": "dog"},
                                   {"tid": 1, "category": "ball"},
                                   {"tid": 2, "category": "car"}],
               "relation_instances": [], "trajectories": traj}
        with open(path, "w") as handle:
            json.dump(doc, handle)
        return path

    def _both_floors(self, path):
        """(clip-wide floor, this window's floor, baseline mse) for window 8.

        Computed here from `oracle` primitives rather than read out of the
        survey, so the test states what the two candidates are instead of
        agreeing with whatever the survey happens to do.
        """
        from tools.planner.common.windows import linear_interp_bboxes
        from tools.planner.box_geometry import boxes_from_vidvrd, round_trip_error

        boxes, _ = boxes_from_vidvrd(path, num_objs=3, fill=False)
        w = self.WINDOW
        mid = list(range(1, w))
        present = (np.abs(boxes).sum(axis=-1) > 0).all(axis=0)
        base = linear_interp_bboxes(boxes[0][present], boxes[w][present],
                                    len(mid))
        e = base - boxes[mid][:, present, :]
        mse = float((e * e).sum(axis=-1).mean())
        return (round_trip_error(boxes, 60, 40),
                round_trip_error(boxes[mid][:, present, :], 60, 40),
                mse)

    def test_the_two_floors_really_do_disagree(self):
        """Guard the fixture: without this the assertions below are vacuous."""
        with tempfile.TemporaryDirectory() as tmp:
            p = self._split_floors(os.path.join(tmp, "FLOORS_0001.json"))
            clip_floor, window_floor, mse = self._both_floors(p)

        self.assertGreater(mse, 0.0)
        self.assertGreater(clip_floor, 10 * window_floor)
        # And they fall on opposite sides of the verdict, so a survey using the
        # wrong one does not merely report a wrong number -- it excludes a clip
        # that is winnable.
        self.assertGreater(clip_floor / mse, 1.0)
        self.assertLess(window_floor / mse, 1.0)

    def test_the_survey_reports_the_windows_own_floor(self):
        from tools.planner.window_survey import survey_clip

        with tempfile.TemporaryDirectory() as tmp:
            p = self._split_floors(os.path.join(tmp, "FLOORS_0001.json"))
            clip_floor, window_floor, mse = self._both_floors(p)
            got = survey_clip(p, 3, 60, 40, fill=False,
                              windows=(self.WINDOW,))

        self.assertIsNotNone(got)
        ratio = got["per_window"][self.WINDOW]
        self.assertAlmostEqual(ratio, window_floor / mse, places=9)
        self.assertNotAlmostEqual(ratio, clip_floor / mse, places=3)

    def test_the_crossover_follows_the_windows_own_floor(self):
        """The number that reaches SPEC V17 is the crossover, so pin it too."""
        from tools.planner.window_survey import survey_clip

        with tempfile.TemporaryDirectory() as tmp:
            p = self._split_floors(os.path.join(tmp, "FLOORS_0001.json"))
            got = survey_clip(p, 3, 60, 40, fill=False,
                              windows=(self.WINDOW,))

        # Under the clip-wide floor this clip's ratio is 1.32 and it is filed
        # as "never crosses". Under its own window's floor it is 0.06.
        self.assertEqual(got["crossover"], self.WINDOW)

    def test_absent_objects_pay_into_neither_side(self):
        """The floor's object set must be the baseline's object set.

        Removing the two blinking objects entirely must not move the reported
        ratio at all: they were never scoreable in this window. Under the
        clip-wide floor, deleting them changed the answer by 20x.
        """
        from tools.planner.window_survey import survey_clip

        with tempfile.TemporaryDirectory() as tmp:
            p = self._split_floors(os.path.join(tmp, "FLOORS_0001.json"))
            with open(p) as handle:
                doc = json.load(handle)
            doc["trajectories"] = [[o for o in ents if o["tid"] == 0]
                                   for ents in doc["trajectories"]]
            doc["subject/objects"] = [{"tid": 0, "category": "dog"}]
            q = os.path.join(tmp, "FLOORS_0002.json")
            with open(q, "w") as handle:
                json.dump(doc, handle)

            three = survey_clip(p, 3, 60, 40, fill=False,
                                windows=(self.WINDOW,))
            alone = survey_clip(q, 3, 60, 40, fill=False,
                                windows=(self.WINDOW,))

        self.assertAlmostEqual(three["per_window"][self.WINDOW],
                               alone["per_window"][self.WINDOW], places=9)


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
        # A diagonal is a straight line, so both must say unwinnable. Without
        # this the test above passes on the verdict the two tools would agree
        # on even if neither computed anything.
        self.assertGreater(screen, 1.0)
        self.assertGreater(ratio, 1.0)

    def _detour(self, path, n=45):
        """One object that leaves the straight line between the endpoints.

        The other branch of the criterion. Checking agreement on an
        unwinnable clip alone tests one side of a verdict, and a criterion
        that returned "unwinnable" for everything would pass it.
        """
        xs = [0., 0., 0., 0., 200., 0., 0., 0., 0.] * (n // 9)
        traj = [[{"tid": 0, "bbox": {"xmin": x + 50.0, "ymin": 100.0,
                                     "xmax": x + 90.0, "ymax": 140.0}}]
                for x in xs]
        doc = {"video_id": "DETOUR_0001", "width": 640, "height": 360,
               "fps": 30, "subject/objects": [{"tid": 0, "category": "dog"}],
               "relation_instances": [], "trajectories": traj}
        with open(path, "w") as handle:
            json.dump(doc, handle)
        return path

    def test_the_two_implementations_agree_on_a_winnable_clip_too(self):
        from screen_vidvrd import window_crossover
        from tools.planner.window_survey import survey_clip

        with tempfile.TemporaryDirectory() as tmp:
            p = self._detour(os.path.join(tmp, "DETOUR_0001.json"))
            with open(p) as f:
                doc = json.load(f)
            screen = window_crossover(self._per_object(doc),
                                      doc["width"], doc["height"], window=8)
            survey = survey_clip(p, 1, 60, 40, fill=False, windows=(8,))

        self.assertIsNotNone(screen)
        self.assertIsNotNone(survey)
        ratio = survey["per_window"].get(8)
        self.assertIsNotNone(ratio)
        self.assertEqual(screen < 1.0, ratio < 1.0,
                         "screen %.4f, survey %.4f" % (screen, ratio))
        self.assertLess(screen, 1.0)
        self.assertLess(ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
