"""Something-Else into the VidVRD format the training builder reads.

Plan `notes/plans/2026-10-03-train-the-rungs.md` step 1.2a. The builder
`model/loaders/vidvrd.py` reads one JSON per clip: `video_id`,
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
                                    os.pardir))

_spec = importlib.util.spec_from_file_location(
    "_convert_se", os.path.join(ROOT, "tools", "datasets",
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

    def test_a_frame_listed_twice_makes_a_clip_ineligible(self):
        # Found 2026-10-03: clips 100029, 48867 and 134969 list 114 frames
        # twice with different boxes, and the builder kept the first copy.
        labels = [_label("a", "0000", 0, 0, 5, 5), _label("hand", "hand", 0, 0, 5, 5)]
        frames = [_frame("1", n, labels) for n in range(1, 40)]
        frames.append(_frame("1", 7, [_label("a", "0000", 1, 1, 6, 6)]))
        self.assertEqual(cse.why_not(frames, max_tracks=3, min_frames=30),
                         "a frame is listed twice")

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


class RelationInstances(unittest.TestCase):
    """The PDDL domain's own statement about a clip, in the VidVRD field.

    Plan notes/plans/2026-10-10-relations-and-boxes.md A2. Something-Else
    ships no per-frame relation. The clip's SSv2 template names a PDDL
    action, that action's precondition and effect state which binary
    predicates it changes, and those hold at the ends of the clip and
    nowhere in between. Measured 2026-10-11 over the 234 clips that train:
    every clip's template maps to a flipping action, 151 argument pairs
    join two objects and 139 join an object to the hand, and the
    placeholders bind to the tracks by position in 227 of 234 clips.
    """

    # `nextto` holds in the precondition and the action never changes it.
    # The field lists what is TRUE, not only what changes, so it belongs in
    # the output. The first version of this code wrote only the flipped
    # predicates and left 98 of the 234 training clips with nothing,
    # measured 2026-10-11.
    DOMAIN = {"put-on": ("(and (not (on ?a ?b)) (in ?a hand)"
                         " (nextto ?a ?b))",
                         "(and (on ?a ?b) (not (in ?a hand)))")}
    PREDICATES = {"on": 2, "in": 2, "nextto": 2, "is-rigid": 1}

    def _doc(self, frames):
        return {"video_id": "c", "width": 100, "height": 100,
                "subject/objects": [{"tid": 0, "category": "cup"},
                                    {"tid": 1, "category": "table"},
                                    {"tid": 2, "category": "hand"}],
                "trajectories": frames}

    def _full(self, tids):
        return [{"tid": t, "bbox": {"xmin": 0, "ymin": 0,
                                    "xmax": 9, "ymax": 9}} for t in tids]

    def test_an_effect_that_turns_true_lands_on_the_last_anchor(self):
        doc = self._doc([self._full([0, 1, 2])] * 3)
        got = cse.relation_instances(doc, ["cup", "table"], "put-on",
                                     self.DOMAIN, self.PREDICATES)
        on = [r for r in got if r["predicate"] == "on"]
        self.assertEqual(on, [{"predicate": "on", "subject_tid": 0,
                               "object_tid": 1, "begin_fid": 2,
                               "end_fid": 3}])

    def test_a_precondition_that_is_true_lands_on_the_first_anchor(self):
        doc = self._doc([self._full([0, 1, 2])] * 3)
        got = cse.relation_instances(doc, ["cup", "table"], "put-on",
                                     self.DOMAIN, self.PREDICATES)
        inn = [r for r in got if r["predicate"] == "in"]
        self.assertEqual(inn, [{"predicate": "in", "subject_tid": 0,
                                "object_tid": 2, "begin_fid": 0,
                                "end_fid": 1}])

    def test_the_anchors_are_the_first_and_last_fully_boxed_frames(self):
        doc = self._doc([self._full([0, 2]), self._full([0, 1, 2]),
                         self._full([0, 1, 2]), self._full([1])])
        got = cse.relation_instances(doc, ["cup", "table"], "put-on",
                                     self.DOMAIN, self.PREDICATES)
        self.assertEqual(sorted(set((r["begin_fid"], r["end_fid"])
                                    for r in got)), [(1, 2), (2, 3)])

    def test_a_clip_with_no_fully_boxed_frame_gets_no_relation(self):
        doc = self._doc([self._full([0]), self._full([1])])
        self.assertEqual(cse.relation_instances(
            doc, ["cup", "table"], "put-on", self.DOMAIN, self.PREDICATES), [])

    def test_the_hand_constant_binds_to_the_hand_track(self):
        doc = self._doc([self._full([0, 1, 2])] * 2)
        got = cse.relation_instances(doc, ["cup", "table"], "put-on",
                                     self.DOMAIN, self.PREDICATES)
        inn = [r for r in got if r["predicate"] == "in"][0]
        self.assertEqual(inn["object_tid"], 2)

    def test_placeholders_bind_to_object_tracks_by_position(self):
        doc = self._doc([self._full([0, 1, 2])] * 2)
        got = cse.relation_instances(doc, ["cup", "table"], "put-on",
                                     self.DOMAIN, self.PREDICATES)
        on = [r for r in got if r["predicate"] == "on"][0]
        self.assertEqual((on["subject_tid"], on["object_tid"]), (0, 1))

    def test_a_true_precondition_that_the_action_never_changes_is_written(self):
        doc = self._doc([self._full([0, 1, 2])] * 3)
        got = cse.relation_instances(doc, ["cup", "table"], "put-on",
                                     self.DOMAIN, self.PREDICATES)
        self.assertIn({"predicate": "nextto", "subject_tid": 0,
                       "object_tid": 1, "begin_fid": 0, "end_fid": 1}, got)

    def test_a_single_argument_predicate_is_not_a_relation(self):
        doc = self._doc([self._full([0, 1, 2])] * 2)
        got = cse.relation_instances(doc, ["cup", "table"], "put-on",
                                     self.DOMAIN, self.PREDICATES)
        self.assertEqual([r for r in got if r["predicate"] == "is-rigid"], [])

    def test_one_assertion_survives_a_negation_of_the_same_predicate(self):
        """`bury` asserts `(in ?a hand)` and negates `(in ?b hand)` and
        `(in ?a ?b)` in the same precondition. Judging the predicate by name
        alone threw the true one away. Found by an adversary 2026-10-11."""
        dom = {"bury": ("(and (in ?a hand) (not (in ?b hand))"
                        " (not (in ?a ?b)))", "(and (in ?a ?b))")}
        doc = self._doc([self._full([0, 1, 2])] * 2)
        got = cse.relation_instances(doc, ["cup", "table"], "bury", dom,
                                     {"in": 2})
        self.assertIn({"predicate": "in", "subject_tid": 0, "object_tid": 2,
                       "begin_fid": 0, "end_fid": 1}, got)

    def test_a_negated_pair_is_not_written_even_when_another_is_asserted(self):
        dom = {"bury": ("(and (in ?a hand) (not (in ?a ?b)))", "(and)")}
        doc = self._doc([self._full([0, 1, 2])] * 2)
        got = cse.relation_instances(doc, ["cup", "table"], "bury", dom,
                                     {"in": 2})
        self.assertEqual([r for r in got
                          if (r["subject_tid"], r["object_tid"]) == (0, 1)],
                         [])

    def test_a_clip_whose_track_count_differs_from_the_placeholders(self):
        doc = {"video_id": "c", "width": 100, "height": 100,
               "subject/objects": [{"tid": 0, "category": "cup"},
                                   {"tid": 1, "category": "hand"}],
               "trajectories": [self._full([0, 1])] * 2}
        self.assertEqual(cse.relation_instances(
            doc, ["cup", "table"], "put-on", self.DOMAIN, self.PREDICATES), [])


class ChooseOnly(unittest.TestCase):
    """An explicit list of clips, so a conversion can be repeated.

    The clips of `se-f6.txt` were chosen by `choose` under the code of
    2026-10-03. A check added that day, which refuses a clip listing one
    frame twice, changed the usable pool from 37,000 to 36,759, and the
    seeded sample of a changed pool shares NOT ONE clip with the old one:
    measured 2026-10-11 on job 47390003. A seed does not make a conversion
    repeatable when the pool moves, so the list does.
    """

    def _records(self, names):
        frame = [{"name": "%s/0001.jpg", "labels": []}]
        return dict((n, frame) for n in names)

    def test_only_keeps_the_named_clips_in_sorted_order(self):
        got = cse.choose(self._records(["a", "b", "c"]), {}, set(), 10,
                         only=["c", "a"])
        self.assertEqual(got, ["a", "c"])

    def test_only_ignores_a_name_that_is_not_in_the_records(self):
        got = cse.choose(self._records(["a"]), {}, set(), 10,
                         only=["a", "missing"])
        self.assertEqual(got, ["a"])

    def test_only_overrides_the_count(self):
        got = cse.choose(self._records(["a", "b", "c"]), {}, set(), 1,
                         only=["a", "b", "c"])
        self.assertEqual(got, ["a", "b", "c"])

    def test_without_only_the_template_filter_still_applies(self):
        got = cse.choose(self._records(["a"]), {"a": "9"}, set(["8"]), 10)
        self.assertEqual(got, [])


if __name__ == "__main__":
    unittest.main()
