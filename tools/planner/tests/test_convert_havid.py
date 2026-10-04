"""HA-ViD into the VidVRD format the training builder reads.

Plan `notes/plans/2026-10-03-train-the-rungs.md` step 1.3a. HA-ViD's object
detection release is COCO, one file per split, read on 2026-10-03 from the
val file: images named `frame_NNNNNN.PNG` numbered across every video of the
split in the order of `ObjectDetection_data_video_info.csv`, every frame of
a boxed video present (18,405 images for 18,405 frames), 1280 by 720, `bbox`
as x, y, width, height, and `attributes.track_id` restarting in each video.

The builder wants one JSON per clip with `trajectories`, one list per frame
of `{tid, bbox{xmin, ymin, xmax, ymax}}`, and frame `fid` at
`<frames>/<clip>/<fid:06d>.jpg`.
"""

import importlib.util
import os
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))

_spec = importlib.util.spec_from_file_location(
    "_convert_havid", os.path.join(ROOT, "tools", "video", "convert_havid.py"))
ch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch)


def _coco():
    cats = [{"id": 3, "name": "Box"}, {"id": 41, "name": "Hand"},
            {"id": 21, "name": "Screw (hex)"}]
    # two videos in one split: V1 has 3 frames, V2 has 2, numbered 0..4
    images = [{"id": k + 1, "file_name": "frame_%06d.PNG" % k}
              for k in range(5)]

    def ann(i, img, cat, track, box):
        return {"id": i, "image_id": img, "category_id": cat, "bbox": box,
                "attributes": {"track_id": track, "keyframe": i == 1}}
    anns = [ann(1, 1, 3, 0, [10, 20, 100, 50]),     # V1 frame 0, Box
            ann(2, 1, 41, 1, [5, 5, 10, 10]),       # V1 frame 0, Hand
            ann(3, 2, 3, 0, [11, 20, 100, 50]),     # V1 frame 1
            ann(4, 4, 41, 0, [0, 0, 8, 8]),         # V2 frame 0, Hand, track 0
            ann(5, 5, 21, 1, [1270, 710, 30, 30])]  # V2 frame 1, off the edge
    return {"categories": cats, "images": images, "annotations": anns}


SPANS = [("V1", 0, 3), ("V2", 3, 2)]


class Spans(unittest.TestCase):

    def test_videos_take_consecutive_global_numbers_in_csv_order(self):
        rows = [("V1.mp4", 3), ("V2.mp4", 2)]
        self.assertEqual(ch.spans(rows), SPANS)


class BoxesByVideo(unittest.TestCase):

    def test_each_box_lands_at_its_video_and_its_own_frame(self):
        got = ch.boxes_by_video(_coco(), SPANS, wanted={"V1", "V2"})
        self.assertEqual(sorted(got["V1"]), [0, 1])
        self.assertEqual(got["V1"][0], [(0, "Box", (10, 20, 110, 70)),
                                        (1, "Hand", (5, 5, 15, 15))])
        self.assertEqual(got["V2"][0], [(0, "Hand", (0, 0, 8, 8))])

    def test_videos_not_wanted_are_not_kept(self):
        got = ch.boxes_by_video(_coco(), SPANS, wanted={"V2"})
        self.assertEqual(sorted(got), ["V2"])


class Windows(unittest.TestCase):

    def test_whole_windows_only_and_evenly_spread(self):
        self.assertEqual(ch.window_starts(1000, size=150, per_video=2),
                         [0, 450])
        self.assertEqual(ch.window_starts(149, size=150, per_video=2), [])

    def test_clip_ids_pad_the_start_so_they_sort_by_time(self):
        self.assertEqual(ch.clip_id("S04A04I01M0", 150), "S04A04I01M0_000150")
        self.assertLess(ch.clip_id("V", 150), ch.clip_id("V", 1050))


class ClipJson(unittest.TestCase):

    def test_a_window_counts_frames_from_zero_and_scales_its_boxes(self):
        boxes = ch.boxes_by_video(_coco(), SPANS, wanted={"V1"})["V1"]
        doc = ch.clip_json("V1_000001", boxes, start=1, size=2,
                           width=640, height=360, scale=0.5)
        self.assertEqual(len(doc["trajectories"]), 2)
        self.assertEqual(doc["trajectories"][0][0]["bbox"],
                         {"xmin": 6, "ymin": 10, "xmax": 56, "ymax": 35})
        self.assertEqual(doc["trajectories"][1], [])
        self.assertEqual((doc["video_id"], doc["width"], doc["height"]),
                         ("V1_000001", 640, 360))

    def test_tracks_take_tids_in_track_order_with_their_category(self):
        boxes = ch.boxes_by_video(_coco(), SPANS, wanted={"V1"})["V1"]
        doc = ch.clip_json("V1_000000", boxes, start=0, size=3,
                           width=640, height=360, scale=0.5)
        self.assertEqual(doc["subject/objects"],
                         [{"tid": 0, "category": "Box"},
                          {"tid": 1, "category": "Hand"}])

    def test_a_box_past_the_frame_edge_is_clipped(self):
        boxes = ch.boxes_by_video(_coco(), SPANS, wanted={"V2"})["V2"]
        doc = ch.clip_json("V2_000000", boxes, start=0, size=2,
                           width=640, height=360, scale=0.5)
        self.assertEqual(doc["trajectories"][1][0]["bbox"],
                         {"xmin": 635, "ymin": 355, "xmax": 640, "ymax": 360})


if __name__ == "__main__":
    unittest.main()
