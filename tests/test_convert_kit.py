"""KIT Bimanual into the VidVRD format the training builder reads.

Plan `notes/plans/2026-10-03-train-the-rungs.md` step 2.3. KIT ships per frame
`2d_objects/frame_N.json`, a list of detections, each with `bounding_box`
`{x, y, w, h}` normalised to the image and `candidates`, the first holding
`class_name` and `certainty`. There is no track id, and one class can appear
twice in a frame. Measured 2026-10-03 on 6,252 boxes of three takes: read as a
corner, 959 boxes run past the frame edge; read as a centre, none do. So
`x, y` is the box centre, as YOLO writes it.
"""

import importlib.util
import os
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir))

_spec = importlib.util.spec_from_file_location(
    "_convert_kit", os.path.join(ROOT, "tools", "video", "convert_kit.py"))
ck = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ck)


def _det(name, certainty, x, y, w, h):
    return {"bounding_box": {"x": x, "y": y, "w": w, "h": h},
            "candidates": [{"class_name": name, "certainty": certainty}]}


class Boxes(unittest.TestCase):

    def test_a_centre_box_becomes_corners_in_pixels(self):
        got = ck.frame_boxes([_det("bowl", 0.9, 0.5, 0.5, 0.25, 0.5)], 640, 480)
        self.assertEqual(got, {"bowl": (0.9, (240, 120, 400, 360))})

    def test_a_repeated_class_keeps_its_most_certain_box_and_counts_the_rest(self):
        dets = [_det("knife", 0.6, 0.2, 0.2, 0.1, 0.1),
                _det("knife", 0.8, 0.7, 0.7, 0.1, 0.1),
                _det("bowl", 0.9, 0.5, 0.5, 0.2, 0.2)]
        counts = {"duplicates": 0}
        got = ck.frame_boxes(dets, 100, 100, counts)
        self.assertEqual(got["knife"], (0.8, (65, 65, 75, 75)))
        self.assertEqual(counts["duplicates"], 1)

    def test_a_box_past_the_edge_is_clipped(self):
        got = ck.frame_boxes([_det("board", 0.9, 0.9, 0.9, 0.4, 0.4)], 100, 100)
        self.assertEqual(got["board"][1], (70, 70, 100, 100))


class Windows(unittest.TestCase):

    def test_every_whole_window_is_taken(self):
        self.assertEqual(ck.window_starts(460, size=150), [0, 150, 300])
        self.assertEqual(ck.window_starts(100, size=150), [])

    def test_clip_ids_sort_by_subject_then_task_then_time(self):
        a = ck.clip_id(1, "task_9_w_sawing", 0, 300)
        b = ck.clip_id(6, "task_1_k_cooking", 0, 0)
        self.assertEqual(a, "s1_task_9_w_sawing_take0_000300")
        self.assertLess(a, b)


class ClipJson(unittest.TestCase):

    def test_classes_take_tids_in_name_order_and_frames_count_from_zero(self):
        frames = {10: {"bowl": (0.9, (1, 2, 3, 4)), "knife": (0.8, (5, 6, 7, 8))},
                  11: {"knife": (0.8, (5, 6, 8, 9))}}
        doc = ck.clip_json("c", frames, start=10, size=3, width=640, height=480)
        self.assertEqual(doc["subject/objects"],
                         [{"tid": 0, "category": "bowl"},
                          {"tid": 1, "category": "knife"}])
        self.assertEqual(len(doc["trajectories"]), 3)
        self.assertEqual(doc["trajectories"][1],
                         [{"tid": 1, "generated": 0,
                           "bbox": {"xmin": 5, "ymin": 6, "xmax": 8, "ymax": 9}}])
        self.assertEqual(doc["trajectories"][2], [])


if __name__ == "__main__":
    unittest.main()
