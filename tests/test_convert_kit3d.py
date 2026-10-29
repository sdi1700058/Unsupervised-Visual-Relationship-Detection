"""KIT Bimanual from `3d_objects`: tracked instances and both hands.

Plan `notes/plans/2026-10-05-dataset-fixes.md` (archived, see workbench/ARCHIVE.md), F3. The author decided on
2026-10-05 to convert KIT from its 3D boxes, because the 2D detections name
one box for each class, misname objects and hold no hand. No camera values
ship, so each take fits its own pinhole camera from 3D box centres to the
centres of the single 2D box of the same class, and a take whose held-out
error is above 10 px is dropped.
"""

import unittest

import numpy as np

from tools.datasets import convert_kit3d as ck


TRUE_Q = np.array([560.0, 540.0, 330.0, 250.0, 0.30, -0.05, 0.02,
                   20.0, 720.0, 0.0])


def _table_points(n, seed=0):
    rng = np.random.RandomState(seed)
    return np.c_[rng.uniform(-400, 400, n), rng.uniform(-820, -600, n),
                 rng.uniform(-1700, -1000, n)]


def _box(x0, x1, y0, y1, z0, z1):
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "z0": z0, "z1": z1}


class Camera(unittest.TestCase):

    def test_a_fit_recovers_the_projection_of_a_known_camera(self):
        X = _table_points(200)
        U = ck.project(ck.camera_matrix(TRUE_Q), X)
        q = ck.fit_camera(X, U)
        err = np.hypot(*(ck.project(ck.camera_matrix(q), X) - U).T)
        self.assertLess(np.median(err), 0.5)

    def test_the_held_out_error_is_small_for_a_known_camera(self):
        X = _table_points(200, seed=1)
        U = ck.project(ck.camera_matrix(TRUE_Q), X)
        self.assertLess(ck.heldout_error(X, U), 0.5)

    def test_the_held_out_error_is_large_when_the_two_halves_disagree(self):
        X = _table_points(200, seed=2)
        U = ck.project(ck.camera_matrix(TRUE_Q), X)
        U[100:] += 40.0
        self.assertGreater(ck.heldout_error(X, U), 10.0)


class Orientation(unittest.TestCase):
    """P and -P give the same pixels, so a fit can return a camera with every
    point behind it. Found 2026-10-05: 47 of 66 converted windows held no
    box, because `projected_box` refused boxes behind such a camera."""

    def test_a_camera_with_the_points_behind_it_is_turned_around(self):
        X = _table_points(50)
        P = -ck.camera_matrix(TRUE_Q)
        Q = ck.oriented(P, X)
        np.testing.assert_allclose(ck.project(Q, X), ck.project(P, X))
        b = _box(-50, 50, -800, -700, -1300, -1200)
        self.assertIsNone(ck.projected_box(P, b, 640, 480))
        self.assertIsNotNone(ck.projected_box(Q, b, 640, 480))


class ProjectedBox(unittest.TestCase):

    def test_the_box_holds_all_eight_projected_corners_inside_the_image(self):
        P = ck.camera_matrix(TRUE_Q)
        b = _box(-50, 50, -800, -700, -1300, -1200)
        x1, y1, x2, y2 = ck.projected_box(P, b, 640, 480)
        corners = ck.project(P, ck.box_corners(b))
        self.assertLessEqual(x1, corners[:, 0].min() + 1)
        self.assertGreaterEqual(x2, corners[:, 0].max() - 1)
        self.assertTrue(0 <= x1 < x2 <= 640 and 0 <= y1 < y2 <= 480)

    def test_a_box_behind_the_camera_gives_none(self):
        P = ck.camera_matrix(TRUE_Q)
        self.assertIsNone(ck.projected_box(P, _box(-50, 50, -800, -700,
                                                   400, 500), 640, 480))

    def test_a_box_wholly_outside_the_image_gives_none(self):
        P = ck.camera_matrix(TRUE_Q)
        self.assertIsNone(ck.projected_box(P, _box(5000, 5100, -800, -700,
                                                   -1300, -1200), 640, 480))


class Matching(unittest.TestCase):

    def test_a_class_with_one_box_in_each_gives_one_pair(self):
        d2 = [{"bounding_box": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
               "candidates": [{"class_name": "cup"}]},
              {"bounding_box": {"x": 0.2, "y": 0.2, "w": 0.1, "h": 0.1},
               "candidates": []}]
        d3 = [{"class_name": "cup", "instance_name": "cup_1",
               "bounding_box": _box(0, 10, 0, 10, -1010, -1000)}]
        pairs = ck.matched_centres(d2, d3, 640, 480)
        self.assertEqual(len(pairs), 1)
        np.testing.assert_allclose(pairs[0][0], [5, 5, -1005])
        np.testing.assert_allclose(pairs[0][1], [320, 240])

    def test_a_class_seen_twice_in_3d_gives_no_pair(self):
        d2 = [{"bounding_box": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
               "candidates": [{"class_name": "bowl"}]}]
        d3 = [{"class_name": "bowl", "instance_name": "bowl_1",
               "bounding_box": _box(0, 10, 0, 10, -1010, -1000)},
              {"class_name": "bowl", "instance_name": "bowl_2",
               "bounding_box": _box(90, 99, 0, 10, -1010, -1000)}]
        self.assertEqual(ck.matched_centres(d2, d3, 640, 480), [])


class Instances(unittest.TestCase):

    def test_an_instance_takes_its_most_frequent_class(self):
        frames = [[{"instance_name": "banana_2", "class_name": c}]
                  for c in ("banana", "knife", "banana", "bowl", "banana")]
        self.assertEqual(ck.instance_classes(frames), {"banana_2": "banana"})

    def test_the_hands_are_named_by_their_instance(self):
        frames = [[{"instance_name": "LeftHand_5", "class_name": "LeftHand"}]]
        self.assertEqual(ck.instance_classes(frames),
                         {"LeftHand_5": "LeftHand"})


class FrameRate(unittest.TestCase):

    def test_the_rate_comes_from_the_metadata_of_the_take(self):
        text = ("name,type,value\nfps,unsigned int,15\n"
                "framesPerChunk,unsigned int,100\n")
        self.assertEqual(ck.take_fps(text), 15)


class ClipJson(unittest.TestCase):

    def test_a_track_is_an_instance_and_its_category_is_the_class(self):
        frames = {0: {"cup_2": (0.9, (1, 2, 30, 40)),
                      "LeftHand_5": (1.0, (50, 60, 70, 80))},
                  1: {"cup_2": (0.9, (2, 3, 31, 41))}}
        doc = ck.clip_json("c", frames, 0, 2, 640, 480,
                           {"cup_2": "cup", "LeftHand_5": "LeftHand"})
        cats = dict((o["tid"], o["category"]) for o in doc["subject/objects"])
        self.assertEqual(sorted(cats.values()), ["LeftHand", "cup"])
        self.assertEqual(len(doc["trajectories"]), 2)
        self.assertEqual(len(doc["trajectories"][1]), 1)



def _rel(frames):
    """Relations of `frames` frames: the right hand touches the cup, and
    the cup touches the bowl."""
    objs = [{"instance_name": n} for n in ("RightHand_4", "LeftHand_5",
                                            "cup_2", "bowl_7", "knife_3")]
    rels = [{"subject_index": 0, "relation_name": "contact",
             "object_index": 2},
            {"subject_index": 2, "relation_name": "moving together",
             "object_index": 0},
            {"subject_index": 2, "relation_name": "contact",
             "object_index": 3},
            {"subject_index": 1, "relation_name": "left of",
             "object_index": 4}]
    return [objs] * frames, [rels] * frames


class Roles(unittest.TestCase):
    """Role-based slot selection for KIT. Plan
    notes/plans/2026-10-05-dataset-fixes.md (archived, see workbench/ARCHIVE.md), F5. The labels give an action
    for each hand and no object (`dreher2019` p.3), so the moved object is
    the one in contact with the acting hand most often, and the target the
    one in contact with the moved object most often, by the
    `spatial_relations` KIT derives from its 3D boxes."""

    def test_a_label_list_becomes_segments(self):
        doc = {"right_hand": [0, 0, 29, 1, 77, 3, 88, 7, 158],
               "left_hand": [0, 0, 158]}
        self.assertEqual(ck.read_actions(doc)["right_hand"],
                         [(0, 28, "idle"), (29, 76, "approach"),
                          (77, 87, "lift"), (88, 157, "pour")])

    def test_an_unlabelled_segment_ends_a_cycle_as_idle_does(self):
        # subject_3/task_2_k_cooking_with_bowls/take_1 labels frames 194 to
        # 198 of the right hand with null
        doc = {"right_hand": [0, 0, 61, 1, 85, 5, 194, None, 199, 1, 206, 3,
                              300]}
        segments = ck.read_actions(doc)["right_hand"]
        self.assertEqual(segments[3], (194, 198, None))
        self.assertEqual(ck.cycles(segments), [(61, 193), (199, 299)])

    def test_a_cycle_runs_from_one_idle_to_the_next(self):
        segments = [(0, 9, "idle"), (10, 19, "approach"), (20, 29, "lift"),
                    (30, 49, "pour"), (50, 54, "retreat"), (55, 60, "idle"),
                    (61, 70, "approach"), (71, 80, "place")]
        self.assertEqual(ck.cycles(segments), [(10, 54), (61, 80)])

    def test_the_hand_its_moved_object_and_the_target(self):
        objs, rels = _rel(10)
        self.assertEqual(ck.kit_role_tracks(objs, rels, 0, 9, "RightHand"),
                         ("RightHand_4", "cup_2", "bowl_7"))

    def test_a_hand_that_touches_nothing_gives_no_roles(self):
        objs, rels = _rel(10)
        self.assertIsNone(ck.kit_role_tracks(objs, rels, 0, 9, "LeftHand"))


class RelationInstances(unittest.TestCase):
    """KIT's own `spatial_relations` into the VidVRD field, plan
    notes/plans/2026-10-10-relations-and-boxes.md (archived, see workbench/ARCHIVE.md) A1. The author chose the
    dataset's own relations on 2026-10-10. `begin_fid` counts from the start
    of the CLIP and `end_fid` is one past the last frame, as VidVRD writes
    them."""

    def _objs(self, order):
        return [{"instance_name": n} for n in order]

    def test_one_run_of_frames_is_one_instance(self):
        order = ["RightHand_4", "cup_2", "bowl_7"]
        d3 = [self._objs(order)] * 3
        rels = [[{"subject_index": 0, "relation_name": "contact",
                  "object_index": 1}]] * 3
        got = ck.relation_instances(d3, rels, ("RightHand_4", "cup_2"),
                                    0, 3, {"RightHand_4": 0, "cup_2": 1})
        self.assertEqual(got, [{"predicate": "contact", "subject_tid": 0,
                                "object_tid": 1, "begin_fid": 0,
                                "end_fid": 3}])

    def test_a_gap_splits_one_run_into_two_instances(self):
        order = ["RightHand_4", "cup_2"]
        on = [{"subject_index": 0, "relation_name": "contact",
               "object_index": 1}]
        d3 = [self._objs(order)] * 4
        got = ck.relation_instances(d3, [on, [], on, on],
                                    ("RightHand_4", "cup_2"), 0, 4,
                                    {"RightHand_4": 0, "cup_2": 1})
        self.assertEqual([(g["begin_fid"], g["end_fid"]) for g in got],
                         [(0, 1), (2, 4)])

    def test_a_relation_that_leaves_the_three_tracks_is_dropped(self):
        order = ["RightHand_4", "cup_2", "knife_3"]
        d3 = [self._objs(order)] * 2
        rels = [[{"subject_index": 1, "relation_name": "left of",
                  "object_index": 2}]] * 2
        self.assertEqual(ck.relation_instances(
            d3, rels, ("RightHand_4", "cup_2"), 0, 2,
            {"RightHand_4": 0, "cup_2": 1}), [])

    def test_the_index_is_read_against_that_frame_and_not_the_first(self):
        # KIT lists the objects of each frame on their own, so index 0 is a
        # different instance in frame 1 than in frame 0.
        d3 = [self._objs(["cup_2", "RightHand_4"]),
              self._objs(["RightHand_4", "cup_2"])]
        rels = [[{"subject_index": 1, "relation_name": "contact",
                  "object_index": 0}],
                [{"subject_index": 0, "relation_name": "contact",
                  "object_index": 1}]]
        got = ck.relation_instances(d3, rels, ("RightHand_4", "cup_2"), 0, 2,
                                    {"RightHand_4": 0, "cup_2": 1})
        self.assertEqual(got, [{"predicate": "contact", "subject_tid": 0,
                                "object_tid": 1, "begin_fid": 0,
                                "end_fid": 2}])

    def test_the_frame_numbers_count_from_the_start_of_the_clip(self):
        # A clip that starts at frame 29 of its take writes begin_fid 0.
        order = ["RightHand_4", "cup_2"]
        d3 = [self._objs(order)] * 31
        rels = [[]] * 29 + [[{"subject_index": 0, "relation_name": "contact",
                              "object_index": 1}]] * 2
        got = ck.relation_instances(d3, rels, ("RightHand_4", "cup_2"), 29, 2,
                                    {"RightHand_4": 0, "cup_2": 1})
        self.assertEqual([(g["begin_fid"], g["end_fid"]) for g in got],
                         [(0, 2)])

    def test_a_negative_index_is_dropped_and_not_read_from_the_end(self):
        # Python reads objs[-1] as the last object, so a bounds check that
        # tests only the upper end pairs the wrong instances in silence.
        # Added 2026-10-10 after an adversary deleted the check and all six
        # tests stayed green.
        order = ["RightHand_4", "cup_2"]
        rels = [[{"subject_index": -1, "relation_name": "contact",
                  "object_index": 0}]] * 2
        self.assertEqual(ck.relation_instances(
            [self._objs(order)] * 2, rels, ("RightHand_4", "cup_2"), 0, 2,
            {"RightHand_4": 0, "cup_2": 1}), [])

    def test_a_window_past_the_end_of_the_take_stops_at_the_end(self):
        # role_clips clamps the window, so this is a guard and not a path
        # the converter reaches. An adversary deleted the guard on
        # 2026-10-10 and all six tests stayed green.
        order = ["RightHand_4", "cup_2"]
        on = [{"subject_index": 0, "relation_name": "contact",
               "object_index": 1}]
        got = ck.relation_instances([self._objs(order)], [on],
                                    ("RightHand_4", "cup_2"), 0, 3,
                                    {"RightHand_4": 0, "cup_2": 1})
        self.assertEqual([(g["begin_fid"], g["end_fid"]) for g in got],
                         [(0, 1)])

    def test_a_frame_without_a_relation_file_breaks_the_run(self):
        order = ["RightHand_4", "cup_2"]
        on = [{"subject_index": 0, "relation_name": "contact",
               "object_index": 1}]
        got = ck.relation_instances([self._objs(order)] * 3, [on, [], on],
                                    ("RightHand_4", "cup_2"), 0, 3,
                                    {"RightHand_4": 0, "cup_2": 1})
        self.assertEqual(len(got), 2)


class TiesAreBroken(unittest.TestCase):
    """A tie in the role counts must not be settled by the hash seed.

    `collections.Counter.most_common` returns equal counts in insertion
    order, and the insertion order of these counters follows a set of
    instance names, which Python orders by hash. An adversary ran one real
    clip under three values of PYTHONHASHSEED on 2026-10-11 and got three
    different target objects: banana_10, cereals_2 and knife_6, all four
    candidates touching the moved object in 80 of 80 frames. Two of the 199
    role clips have such a tie. The name settles it now, so a conversion
    repeats.
    """

    def _frames(self, n, extra_names):
        objs = [{"instance_name": x} for x in
                ["RightHand_4", "cup_2"] + extra_names]
        rels = [{"subject_index": 0, "relation_name": "contact",
                 "object_index": 1}]
        # every extra object touches the moved object in every frame
        for k in range(len(extra_names)):
            rels.append({"subject_index": 1, "relation_name": "contact",
                         "object_index": 2 + k})
        return [objs] * n, [rels] * n

    def test_a_tie_on_the_target_goes_to_the_first_name(self):
        objs, rels = self._frames(10, ["zebra_9", "apple_3"])
        self.assertEqual(
            ck.kit_role_tracks(objs, rels, 0, 9, "RightHand")[2], "apple_3")

    def test_the_tie_does_not_depend_on_the_order_of_the_objects(self):
        o1, r1 = self._frames(10, ["zebra_9", "apple_3"])
        o2, r2 = self._frames(10, ["apple_3", "zebra_9"])
        a = ck.kit_role_tracks(o1, r1, 0, 9, "RightHand")
        b = ck.kit_role_tracks(o2, r2, 0, 9, "RightHand")
        self.assertEqual(a[2], b[2])


if __name__ == "__main__":
    unittest.main()
