"""The ceiling for the thesis question: perfect relations, fed to a planner.

**Why this exists, and why the other oracle is not it.** `oracle.py` builds a
planner state from ground-truth *boxes*. That answers "can a classical planner
do frame interpolation from a binary state at all", which is a ceiling on the
planner. It says nothing about relations, and the thesis asks whether FOSAE's
*relations* are plannable.

Ground-truth relations were never fed to a planner anywhere in this project.
`relation_instances` was read by exactly two files, `compositional.py` and
`m7_map.py`, and both are scorers.

An oracle is a **source of state**, not a metric. This one is a drop-in
replacement for the FOSAE latent, built from the annotations, so every existing
method can run on it unchanged. If a planner cannot use perfect relations, no
amount of training would have helped.

The VidVRD annotation gives, per relation:

    {"subject_tid": 0, "predicate": "stand_behind", "object_tid": 1,
     "begin_fid": 0, "end_fid": 30}

Verified against the data: `end_fid` is **exclusive**, and never exceeds the
trajectory length. Relations are annotated in 30-frame segments.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
    os.pardir)))

from tools.planner import relation_oracle as ro


def clip(n_frames=6, rels=None, objs=None):
    """A minimal annotation in the real VidVRD shape."""
    objs = objs if objs is not None else [{"tid": 0, "category": "dog"},
                                          {"tid": 1, "category": "frisbee"}]
    boxes = []
    for f in range(n_frames):
        boxes.append([{"tid": o["tid"],
                       "bbox": {"xmin": 10 * (i + 1) + f, "ymin": 10,
                                "xmax": 10 * (i + 1) + f + 20, "ymax": 40}}
                      for i, o in enumerate(objs)])
    return {"video_id": "v", "width": 640, "height": 480, "fps": 30,
            "frame_count": n_frames, "trajectories": boxes,
            "subject/objects": objs,
            "relation_instances": rels if rels is not None else []}


def rel(s, p, o, a, b):
    return {"subject_tid": s, "predicate": p, "object_tid": o,
            "begin_fid": a, "end_fid": b}


class TestVocabulary(unittest.TestCase):

    def test_the_predicates_of_the_clip_become_the_vocabulary(self):
        doc = clip(rels=[rel(0, "bite", 1, 0, 3), rel(0, "hold", 1, 1, 4)])
        self.assertEqual(ro.vocabulary(doc), ["bite", "hold"])

    def test_it_is_sorted_so_the_layout_is_reproducible(self):
        doc = clip(rels=[rel(0, "hold", 1, 0, 3), rel(0, "bite", 1, 0, 3)])
        self.assertEqual(ro.vocabulary(doc), ["bite", "hold"])

    def test_a_repeated_predicate_appears_once(self):
        doc = clip(rels=[rel(0, "bite", 1, 0, 3), rel(1, "bite", 0, 0, 3)])
        self.assertEqual(ro.vocabulary(doc), ["bite"])

    def test_a_clip_with_no_relations_has_an_empty_vocabulary(self):
        self.assertEqual(ro.vocabulary(clip()), [])


class TestTheState(unittest.TestCase):

    def _states(self, doc, num_objs=2, vocab=None):
        return ro.states_from_annotation(doc, num_objs=num_objs,
                                         vocab=vocab)

    def test_a_relation_is_set_exactly_on_the_frames_it_covers(self):
        """`end_fid` is exclusive. Verified against the real annotations."""
        doc = clip(6, [rel(0, "bite", 1, 2, 4)])
        st, _, _ = self._states(doc)
        on = [f for f in range(6) if st[f].any()]
        self.assertEqual(on, [2, 3])

    def test_a_frame_outside_every_relation_is_all_zero(self):
        doc = clip(4, [rel(0, "bite", 1, 0, 2)])
        st, _, _ = self._states(doc)
        self.assertEqual(int(st[3].sum()), 0)

    def test_direction_matters(self):
        """`dog bites frisbee` is not `frisbee bites dog`."""
        a, _, _ = self._states(clip(3, [rel(0, "bite", 1, 0, 3)]))
        b, _, _ = self._states(clip(3, [rel(1, "bite", 0, 0, 3)]))
        self.assertFalse(np.array_equal(a[0], b[0]))
        self.assertEqual(int(a[0].sum()), int(b[0].sum()))

    def test_two_relations_on_one_frame_set_two_bits(self):
        doc = clip(3, [rel(0, "bite", 1, 0, 3), rel(0, "hold", 1, 0, 3)])
        st, _, _ = self._states(doc)
        self.assertEqual(int(st[0].sum()), 2)

    def test_the_width_is_ordered_pairs_times_predicates(self):
        doc = clip(3, [rel(0, "bite", 1, 0, 3), rel(0, "hold", 1, 0, 3)])
        st, vocab, _ = self._states(doc, num_objs=3)
        self.assertEqual(len(vocab), 2)
        self.assertEqual(st.shape[1], 3 * 2 * 2)      # 6 ordered pairs x 2

    def test_an_object_is_never_related_to_itself(self):
        """Self pairs are not in the layout, so no bit can encode one."""
        self.assertEqual(ro.pair_count(3), 6)
        self.assertEqual(ro.pair_count(2), 2)
        self.assertEqual(ro.pair_count(1), 0)

    def test_the_state_is_binary(self):
        doc = clip(3, [rel(0, "bite", 1, 0, 3)])
        st, _, _ = self._states(doc)
        self.assertEqual(sorted(set(st.reshape(-1).tolist())), [0, 1])

    def test_a_relation_naming_an_absent_tid_is_dropped_not_guessed(self):
        """A tid outside the slot map has no slot, so it cannot be placed."""
        doc = clip(3, [rel(0, "bite", 99, 0, 3)])
        st, _, stats = self._states(doc)
        self.assertEqual(int(st.sum()), 0)
        self.assertEqual(stats["dropped_unmapped"], 1)

    def test_relations_beyond_the_slot_count_are_counted(self):
        """With num_objs=2 a third object's relations cannot be represented."""
        objs = [{"tid": i, "category": "o%d" % i} for i in range(3)]
        doc = clip(3, [rel(0, "bite", 2, 0, 3)], objs=objs)
        _, _, stats = self._states(doc, num_objs=2)
        self.assertGreater(stats["dropped_unmapped"], 0)


class TestSlotsAreStable(unittest.TestCase):

    def test_one_tid_keeps_one_slot_for_the_whole_clip(self):
        doc = clip(5, [rel(0, "bite", 1, 0, 5)])
        st, _, _ = ro.states_from_annotation(doc, num_objs=2)
        for f in range(1, 5):
            self.assertTrue(np.array_equal(st[0], st[f]))

    def test_the_slot_map_is_recorded(self):
        doc = clip(3, [rel(0, "bite", 1, 0, 3)])
        _, _, stats = ro.states_from_annotation(doc, num_objs=2)
        self.assertIn("slot_of_tid", stats)
        self.assertEqual(len(stats["slot_of_tid"]), 2)


class TestItIsPlannable(unittest.TestCase):
    """A planner mines actions from the deltas between consecutive states."""

    def test_a_relation_starting_midway_gives_a_transition(self):
        doc = clip(4, [rel(0, "bite", 1, 2, 4)])
        st, _, _ = ro.states_from_annotation(doc, num_objs=2)
        deltas = [int((st[i] != st[i + 1]).sum()) for i in range(3)]
        self.assertEqual(deltas, [0, 1, 0])

    def test_a_clip_where_nothing_changes_has_no_transition(self):
        """Reported, because such a clip teaches a planner nothing."""
        doc = clip(4, [rel(0, "bite", 1, 0, 4)])
        st, _, _ = ro.states_from_annotation(doc, num_objs=2)
        self.assertTrue(all(int((st[i] != st[i + 1]).sum()) == 0
                            for i in range(3)))


class TestTheExport(unittest.TestCase):
    """It has to be an ordinary export, or no existing method can read it."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, doc, **kw):
        src = os.path.join(self.dir, "v.json")
        with open(src, "w") as handle:
            json.dump(doc, handle)
        out = os.path.join(self.dir, "out.npz")
        ro.build_export(src, out, num_objs=2, **kw)
        return np.load(out, allow_pickle=False)

    def test_it_carries_the_keys_every_reader_expects(self):
        d = self._write(clip(4, [rel(0, "bite", 1, 1, 3)]))
        for key in ("latents", "gt_boxes", "decoded_boxes", "n_bits",
                    "model_name", "frame_ids"):
            self.assertIn(key, d.files)

    def test_the_boxes_are_the_real_ones_so_scoring_still_works(self):
        """The relational state plans; the boxes score. Both are ground truth."""
        d = self._write(clip(4, [rel(0, "bite", 1, 1, 3)]))
        self.assertEqual(d["gt_boxes"].shape, (4, 2, 4))
        self.assertTrue(np.array_equal(d["gt_boxes"], d["decoded_boxes"]))

    def test_n_bits_matches_the_latent_width(self):
        d = self._write(clip(4, [rel(0, "bite", 1, 1, 3)]))
        self.assertEqual(int(d["n_bits"]), d["latents"].shape[1])

    def test_the_vocabulary_travels_with_the_export(self):
        """Without it the latent is an uninterpretable bit vector."""
        d = self._write(clip(4, [rel(0, "bite", 1, 1, 3),
                                 rel(0, "hold", 1, 0, 4)]))
        self.assertEqual([str(p) for p in d["predicates"]], ["bite", "hold"])

    def test_it_says_what_it_is(self):
        d = self._write(clip(4, [rel(0, "bite", 1, 1, 3)]))
        self.assertIn("relation", str(d["model_name"]).lower())

    def test_a_clip_with_no_relations_refuses_rather_than_writing_zeros(self):
        """An all-zero export would read as a planner that solved nothing."""
        src = os.path.join(self.dir, "empty.json")
        with open(src, "w") as handle:
            json.dump(clip(4), handle)
        self.assertRaises(SystemExit, ro.build_export, src,
                          os.path.join(self.dir, "o.npz"), num_objs=2)


if __name__ == "__main__":
    unittest.main()
