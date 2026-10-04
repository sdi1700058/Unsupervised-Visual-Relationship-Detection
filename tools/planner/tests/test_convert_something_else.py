"""Something-Else into the VidVRD format the training builder reads.

Plan `notes/plans/2026-10-03-train-the-rungs.md` step 1.2a. The builder
`latplan/puzzles/puzzle_vidvrd.py` reads one JSON per clip: `video_id`,
`width`, `height`, `subject/objects` of `{tid, category}` and `trajectories`,
one list per frame of `{tid, bbox{xmin, ymin, xmax, ymax}}`, with frame `fid`
at `<frames>/<video_id>/<fid:06d>.jpg`.

Something-Else gives, per video, a list of annotated frames named
`<video>/<NNNN>.jpg` counted from 1, each with labels carrying `box2d`, a
free-text `category` and a `standard_category` that is a per-video track id:
`0000`, `0001`, `hand`. Read from the head of
`bounding_box_smthsmth_part2.json` on 2026-10-03. That file's first video
starts at `0009.jpg`, so an annotation can cover part of a video only.
"""

import importlib.util
import os
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))

_spec = importlib.util.spec_from_file_location(
    "_convert_se", os.path.join(ROOT, "tools", "video",
                                "convert_something_else.py"))
cse = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cse)


def _label(cat, std, x1, y1, x2, y2):
    return {"box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "category": cat, "standard_category": std}


def _frame(video, n, labels):
    return {"name": "%s/%04d.jpg" % (video, n), "labels": labels}


class FrameIndex(unittest.TestCase):

    def test_frame_names_count_from_one_and_fids_from_zero(self):
        self.assertEqual(cse.frame_index("125757/0009.jpg"), 8)
        self.assertEqual(cse.frame_index("7/0001.jpg"), 0)


class ClipJson(unittest.TestCase):

    def _frames(self):
        return [
            _frame("42", 2, [_label("hand", "hand", 0, 0, 10, 10),
                             _label("pen", "0000", 5.4, 6.6, 50.2, 60.9)]),
            _frame("42", 3, [_label("pen", "0000", 6, 7, 51, 61),
                             _label("cup", "0001", 100, 20, 140, 90)]),
        ]

    def test_objects_take_tids_in_id_order_and_the_hand_last(self):
        doc = cse.clip_json("42", self._frames(), 427, 240)
        self.assertEqual(doc["subject/objects"],
                         [{"tid": 0, "category": "pen"},
                          {"tid": 1, "category": "cup"},
                          {"tid": 2, "category": "hand"}])

    def test_a_frame_lands_at_its_own_index_and_unannotated_frames_are_empty(self):
        doc = cse.clip_json("42", self._frames(), 427, 240)
        self.assertEqual(len(doc["trajectories"]), 3)
        self.assertEqual(doc["trajectories"][0], [])
        self.assertEqual(sorted(o["tid"] for o in doc["trajectories"][1]),
                         [0, 2])

    def test_boxes_become_whole_pixels_inside_the_frame(self):
        frames = [_frame("9", 1, [_label("pen", "0000", -3.2, 5.5, 431.7,
                                         241.0)])]
        doc = cse.clip_json("9", frames, 427, 240)
        self.assertEqual(doc["trajectories"][0][0]["bbox"],
                         {"xmin": 0, "ymin": 6, "xmax": 427, "ymax": 240})

    def test_the_header_names_the_video_and_its_size(self):
        doc = cse.clip_json("42", self._frames(), 427, 240)
        self.assertEqual((doc["video_id"], doc["width"], doc["height"]),
                         ("42", 427, 240))
        self.assertEqual(doc["relation_instances"], [])


class Eligible(unittest.TestCase):

    def test_two_hands_in_one_frame_make_a_clip_ineligible(self):
        frames = [_frame("1", n, [_label("hand", "hand", 0, 0, 5, 5),
                                  _label("hand", "hand", 9, 9, 20, 20)])
                  for n in range(1, 40)]
        self.assertEqual(cse.why_not(frames, max_tracks=3, min_frames=30),
                         "a track id repeats inside one frame")

    def test_more_than_three_tracks_make_a_clip_ineligible(self):
        labels = [_label("a", "0000", 0, 0, 5, 5), _label("b", "0001", 0, 0, 5, 5),
                  _label("c", "0002", 0, 0, 5, 5), _label("hand", "hand", 0, 0, 5, 5)]
        frames = [_frame("1", n, labels) for n in range(1, 40)]
        self.assertEqual(cse.why_not(frames, max_tracks=3, min_frames=30),
                         "more than 3 tracks")

    def test_a_short_clip_is_ineligible_and_a_good_one_is_not(self):
        labels = [_label("a", "0000", 0, 0, 5, 5), _label("hand", "hand", 0, 0, 5, 5)]
        short = [_frame("1", n, labels) for n in range(1, 10)]
        good = [_frame("1", n, labels) for n in range(1, 40)]
        self.assertEqual(cse.why_not(short, max_tracks=3, min_frames=30),
                         "fewer than 30 annotated frames")
        self.assertIsNone(cse.why_not(good, max_tracks=3, min_frames=30))


class Choose(unittest.TestCase):

    def test_only_allowed_templates_are_drawn_and_the_draw_repeats(self):
        labels = [_label("a", "0000", 0, 0, 5, 5), _label("hand", "hand", 0, 0, 5, 5)]
        good = [_frame("v", n, labels) for n in range(1, 40)]
        records = dict((str(v), good) for v in range(20))
        template_of = dict((str(v), "1" if v % 2 else "2") for v in range(20))
        first = cse.choose(records, template_of, {"1"}, count=5, seed=0)
        again = cse.choose(records, template_of, {"1"}, count=5, seed=0)
        self.assertEqual(first, again)
        self.assertEqual(len(first), 5)
        self.assertTrue(all(int(v) % 2 for v in first))


if __name__ == "__main__":
    unittest.main()
