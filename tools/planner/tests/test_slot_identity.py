"""A slot must mean the same object in both frames of a transition.

Both video loaders assigned object slots by descending bbox area, per frame and
independently. When an object leaves the frame, every object behind it moves up
one slot and the vacated tail becomes padding. FOSAE learns its predicates on
slot positions, so if slot 1 holds the dog in frame `i` and the frisbee in frame
`i+1`, the transition it trains on is not the transition that happened.

Measured on the VidVRD production read: **138 adjacent pairs change the slot
name order and 15 change the object count.**

VidVRD ships `tid`, a per-clip track id, and the loader used it only to look up
a category name. Action Genome ships no instance id at all -- verified against
`object_bbox_and_relationship.pkl`, 288,782 frames, whose object dicts carry
`class`, `bbox`, `visible`, three relationship lists and `metadata`, and whose
`metadata['tag']` is `<vid>/<class>/<frame>`. **No class appears twice in any
frame of the 288,782**, so within a video the class name *is* the instance key.

Decided 2026-09-07, the same shape as the adjacency fix of the day before:
**count always, pin only under `STRICT_ADJACENCY=1`.** The default keeps the
area order, so every existing run reproduces; the flag pins each identity to one
slot for the whole clip.

Neither loader can be imported normally here -- `latplan/__init__.py` pulls in
TensorFlow and this environment has none -- so both are loaded from file with
`latplan` and `PIL` stubbed, the approach `test_cache_keys.py` takes. The stubs
come back out in `tearDown`; leaving them in `sys.modules` once turned eleven
tests in three other files red.
"""

import importlib.util
import json
import os
import pickle
import shutil
import sys
import tempfile
import types
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))

_STUBBED = ("latplan", "latplan.util", "latplan.puzzles", "latplan.domains",
            "latplan.domains.video", "latplan.util.paths",
            "latplan.puzzles.puzzle_labeled_objects", "latplan.util.cache",
            "latplan.util.adjacency", "latplan.domains.video.actiongenome",
            "latplan.puzzles.puzzle_vidvrd", "PIL", "PIL.Image")


def _from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Img(object):
    """Enough of a PIL image for either loader: `.convert` and `.size`."""

    size = (100, 100)

    def convert(self, mode):
        return self


def _stub_latplan(tmp):
    import numpy as np

    for name in ("latplan", "latplan.util", "latplan.puzzles",
                 "latplan.domains", "latplan.domains.video"):
        sys.modules.setdefault(name, types.ModuleType(name))

    paths = types.ModuleType("latplan.util.paths")
    paths.DATA_DIR = os.path.join(tmp, "data")
    sys.modules["latplan.util.paths"] = paths

    plo = types.ModuleType("latplan.puzzles.puzzle_labeled_objects")
    plo.PATCH_SIZE = 4
    plo.MAX_OBJECTS = 3
    plo.CANVAS_H, plo.CANVAS_W = 200, 300
    plo.PICSIZE = (200, 300)
    plo._crop_object = lambda img, bbox, patch_size=4: np.full(
        (patch_size, patch_size, 3), 7, dtype=np.uint8)
    plo._scale_bbox_to_canvas = lambda bbox, W, H: tuple(int(v) for v in bbox)
    sys.modules["latplan.puzzles.puzzle_labeled_objects"] = plo

    pil = types.ModuleType("PIL")
    image = types.ModuleType("PIL.Image")
    image.open = lambda path: _Img()
    pil.Image = image
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = image

    _from_file("latplan.util.cache",
               os.path.join(ROOT, "latplan", "util", "cache.py"))
    _from_file("latplan.util.adjacency",
               os.path.join(ROOT, "latplan", "util", "adjacency.py"))


class LoaderCase(unittest.TestCase):
    """Stubs in, temp dataset on disk, both taken away again."""

    def setUp(self):
        self._saved = dict((k, sys.modules.get(k)) for k in _STUBBED)
        self._was_strict = os.environ.get("STRICT_ADJACENCY")
        os.environ.pop("STRICT_ADJACENCY", None)
        self.tmp = tempfile.mkdtemp(prefix="slotid-")
        _stub_latplan(self.tmp)

    def tearDown(self):
        for name, was in self._saved.items():
            if was is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = was
        if self._was_strict is None:
            os.environ.pop("STRICT_ADJACENCY", None)
        else:
            os.environ["STRICT_ADJACENCY"] = self._was_strict
        shutil.rmtree(self.tmp, ignore_errors=True)

    def strict(self, on=True):
        os.environ["STRICT_ADJACENCY"] = "1" if on else "0"


# --------------------------------------------------------------------------
# VidVRD
# --------------------------------------------------------------------------

def _box(x, y, side):
    return {"xmin": x, "ymin": y, "xmax": x + side, "ymax": y + side}


class VidVRDCase(LoaderCase):
    """One clip, three tracks, and the biggest one leaves halfway through."""

    VIDEO = "ILSVRC2015_train_00000001"

    def build_vidvrd(self, trajectories, subjects=None, video_id=None):
        video_id = video_id or self.VIDEO
        ann_dir = os.path.join(self.tmp, "ann", "train")
        frm_dir = os.path.join(self.tmp, "frm", "train", video_id)
        os.makedirs(ann_dir)
        if not os.path.exists(frm_dir):
            os.makedirs(frm_dir)
        if subjects is None:
            subjects = [{"tid": 0, "category": "dog"},
                        {"tid": 1, "category": "frisbee"},
                        {"tid": 2, "category": "person"}]
        with open(os.path.join(ann_dir, video_id + ".json"), "w") as f:
            json.dump({"video_id": video_id, "width": 100, "height": 100,
                       "subject/objects": subjects,
                       "trajectories": trajectories}, f)
        for fid in range(len(trajectories)):
            open(os.path.join(frm_dir, "%06d.jpg" % fid), "w").close()
        return (os.path.join(self.tmp, "ann"), os.path.join(self.tmp, "frm"))

    def loader(self):
        return _from_file("latplan.puzzles.puzzle_vidvrd",
                          os.path.join(ROOT, "latplan", "puzzles",
                                       "puzzle_vidvrd.py"))

    def load(self, trajectories, **kwargs):
        ann_root, frm_root = self.build_vidvrd(trajectories)
        mod = self.loader()
        images, bboxes, names, frame_ids = mod.build_dataset(
            annotations_dir=os.path.join(ann_root, "train"),
            frames_dir=os.path.join(frm_root, "train"),
            num_objs=3, patch_size=4, **kwargs)
        return mod, names

    def three_tracks_one_leaves(self):
        """dog is the largest; it is absent from the middle frame."""
        dog = {"tid": 0, "bbox": _box(0, 0, 90)}
        fris = {"tid": 1, "bbox": _box(0, 0, 20)}
        per = {"tid": 2, "bbox": _box(0, 0, 10)}
        return [[dog, fris, per], [fris, per], [dog, fris, per]]


class TestVidVRDDefaultIsUnchanged(VidVRDCase):
    """The old behaviour, written down so a regression is visible."""

    def test_area_order_when_everything_is_present(self):
        _, names = self.load(self.three_tracks_one_leaves())
        self.assertEqual(names[0], ["dog", "frisbee", "person"])

    def test_a_departure_shifts_every_later_object_up_one_slot(self):
        """This is the defect. It stays the default so old runs reproduce."""
        _, names = self.load(self.three_tracks_one_leaves())
        self.assertEqual(names[1], ["frisbee", "person", "pad_2"])

    def test_slot_one_holds_two_different_objects_across_the_transition(self):
        _, names = self.load(self.three_tracks_one_leaves())
        self.assertEqual(names[0][1], "frisbee")
        self.assertEqual(names[1][1], "person")


class TestVidVRDStrictPinsByTid(VidVRDCase):

    def test_a_departure_leaves_a_hole_and_moves_nobody(self):
        self.strict()
        _, names = self.load(self.three_tracks_one_leaves())
        self.assertEqual(names[1], ["pad_0", "frisbee", "person"])

    def test_every_frame_gives_a_tid_the_same_slot(self):
        self.strict()
        _, names = self.load(self.three_tracks_one_leaves())
        for row in names:
            self.assertEqual(row[1], "frisbee")
            self.assertEqual(row[2], "person")

    def test_the_present_frames_keep_the_order_the_default_gave(self):
        """Pinning changes what a vacancy does, not what a full frame looks
        like: the ranking is by total area, the same quantity the per-frame
        sort used."""
        self.strict()
        _, names = self.load(self.three_tracks_one_leaves())
        self.assertEqual(names[0], ["dog", "frisbee", "person"])
        self.assertEqual(names[2], ["dog", "frisbee", "person"])

    def test_a_track_beyond_the_slot_count_is_dropped_not_rotated(self):
        """Four tracks into three slots: the fourth never appears, in any
        frame, rather than appearing whenever it happens to be large."""
        self.strict()
        objs = [{"tid": 0, "bbox": _box(0, 0, 90)},
                {"tid": 1, "bbox": _box(0, 0, 20)},
                {"tid": 2, "bbox": _box(0, 0, 10)},
                {"tid": 3, "bbox": _box(0, 0, 5)}]
        subjects = [{"tid": 0, "category": "dog"},
                    {"tid": 1, "category": "frisbee"},
                    {"tid": 2, "category": "person"},
                    {"tid": 3, "category": "ball"}]
        ann_root, frm_root = self.build_vidvrd([objs, objs], subjects=subjects)
        mod = self.loader()
        _, _, names, _ = mod.build_dataset(
            annotations_dir=os.path.join(ann_root, "train"),
            frames_dir=os.path.join(frm_root, "train"),
            num_objs=3, patch_size=4)
        for row in names:
            self.assertNotIn("ball", row)
        self.assertEqual(mod.last_load_metadata["slots"]["dropped_past_slots"], 2)


class TestVidVRDMissingTid(VidVRDCase):
    """No id, no pin. Fall back to area order and say how often."""

    def frames_without_tids(self):
        big = {"bbox": _box(0, 0, 90)}
        small = {"tid": 1, "bbox": _box(0, 0, 20)}
        return [[big, small], [big, small]]

    def test_an_object_without_a_tid_is_still_loaded(self):
        self.strict()
        _, names = self.load(self.frames_without_tids())
        self.assertIn("frisbee", names[0])

    def test_the_fallback_is_counted_rather_than_silent(self):
        self.strict()
        mod, _ = self.load(self.frames_without_tids())
        self.assertEqual(mod.last_load_metadata["slots"]["placed_by_area"], 2)

    def test_the_fallback_is_printed(self):
        """A count nobody reads is the same disease as no count."""
        self.strict()
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.load(self.frames_without_tids())
        self.assertIn("slots", buf.getvalue())
        self.assertIn("area", buf.getvalue())


class TestVidVRDMetadata(VidVRDCase):
    """Same shelf as `dropped` and `transitions`, added the day before."""

    def test_the_counts_sit_beside_dropped(self):
        mod, _ = self.load(self.three_tracks_one_leaves())
        meta = mod.last_load_metadata
        self.assertIn("dropped", meta)
        self.assertIn("slots", meta)

    def test_the_record_names_the_mode_and_the_id_it_used(self):
        self.strict()
        mod, _ = self.load(self.three_tracks_one_leaves())
        slots = mod.last_load_metadata["slots"]
        self.assertTrue(slots["strict"])
        self.assertEqual(slots["id_source"], "tid")

    def test_the_default_records_that_it_did_not_pin(self):
        mod, _ = self.load(self.three_tracks_one_leaves())
        self.assertFalse(mod.last_load_metadata["slots"]["strict"])

    def test_the_widest_video_is_recorded_so_a_small_num_objs_is_visible(self):
        mod, _ = self.load(self.three_tracks_one_leaves())
        self.assertEqual(
            mod.last_load_metadata["slots"]["max_identities_in_a_video"], 3)


class TestVidVRDCacheKey(VidVRDCase):
    """Two orderings are two datasets, so they cannot share one npz.

    `cache.py` says a key must name everything that changes what it holds.
    Pinning changes the arrays, so a strict bake reading a default cache would
    train on area-ordered slots while reporting that it pinned them.
    """

    def _cache_path_for(self, strict):
        self.strict(strict)
        ann_root, frm_root = self.build_vidvrd(self.three_tracks_one_leaves())
        mod = self.loader()
        seen = []
        real = mod.save_cache
        mod.save_cache = lambda path, *a, **k: seen.append(path)
        try:
            mod.build_dataset(
                annotations_dir=os.path.join(ann_root, "train"),
                frames_dir=os.path.join(frm_root, "train"),
                num_objs=3, patch_size=4, category_filter="frisbee")
        finally:
            mod.save_cache = real
        self.assertTrue(seen, "build_dataset never wrote a cache")
        return seen[0]

    def test_pinned_and_unpinned_are_two_files(self):
        loose = self._cache_path_for(False)
        self.tearDown()
        self.setUp()
        tight = self._cache_path_for(True)
        self.assertNotEqual(os.path.basename(loose), os.path.basename(tight))

    def test_the_default_keeps_the_name_an_existing_cache_already_has(self):
        self.assertTrue(
            self._cache_path_for(False).endswith("frisbee-3fps-mo3-p4.npz"))


# --------------------------------------------------------------------------
# Action Genome
# --------------------------------------------------------------------------

class ActionGenomeCase(LoaderCase):
    """AG has no track id. The class name is the key, and that is checked."""

    VIDEO = "001YG.mp4"

    def build_ag(self, per_frame, person_frames=None):
        """`per_frame`: list of list of (class, side). Frame n is `00000n.png`."""
        ann_dir = os.path.join(self.tmp, "ag", "annotations")
        frm_dir = os.path.join(self.tmp, "ag", "frames", self.VIDEO)
        os.makedirs(ann_dir)
        os.makedirs(frm_dir)
        obj, per, keys = {}, {}, []
        for n, objs in enumerate(per_frame):
            key = "%s/%06d.png" % (self.VIDEO, n)
            keys.append(key)
            obj[key] = [{"class": cls,
                         "bbox": (0, 0, side, side),
                         "visible": True,
                         "metadata": {"tag": "%s/%s/%06d" % (self.VIDEO, cls, n),
                                      "set": "train"}}
                        for cls, side in objs]
            if person_frames is None or n in person_frames:
                per[key] = {"bbox": [[0, 0, 50, 50]]}
            else:
                per[key] = {"bbox": []}
            open(os.path.join(frm_dir, "%06d.png" % n), "w").close()
        with open(os.path.join(ann_dir, "object_bbox_and_relationship.pkl"), "wb") as f:
            pickle.dump(obj, f)
        with open(os.path.join(ann_dir, "person_bbox.pkl"), "wb") as f:
            pickle.dump(per, f)
        with open(os.path.join(ann_dir, "frame_list.txt"), "w") as f:
            f.write("\n".join(keys) + "\n")
        return ann_dir, os.path.join(self.tmp, "ag", "frames")

    def loader(self):
        return _from_file("latplan.domains.video.actiongenome",
                          os.path.join(ROOT, "latplan", "domains", "video",
                                       "actiongenome.py"))

    def load(self, per_frame, **kwargs):
        ann_dir, frm_dir = self.build_ag(per_frame)
        mod = self.loader()
        _, _, names, _ = mod.build_dataset(annotations_dir=ann_dir,
                                           frames_dir=frm_dir,
                                           num_objs=3, patch_size=4, **kwargs)
        return mod, names

    def table_leaves(self):
        return [[("table", 90), ("cup/glass/bottle", 20)],
                [("cup/glass/bottle", 20)],
                [("table", 90), ("cup/glass/bottle", 20)]]


class TestActionGenomeDefaultIsUnchanged(ActionGenomeCase):

    def test_person_first_then_area_order(self):
        _, names = self.load(self.table_leaves())
        self.assertEqual(names[0], ["person", "table", "cup/glass/bottle"])

    def test_a_departure_shifts_the_rest_up(self):
        _, names = self.load(self.table_leaves())
        self.assertEqual(names[1], ["person", "cup/glass/bottle", "pad_2"])


class TestActionGenomeStrictPinsByClass(ActionGenomeCase):

    def test_a_departure_leaves_a_hole(self):
        self.strict()
        _, names = self.load(self.table_leaves())
        self.assertEqual(names[1], ["person", "pad_1", "cup/glass/bottle"])

    def test_the_class_keeps_its_slot_in_every_frame(self):
        self.strict()
        _, names = self.load(self.table_leaves())
        for row in names:
            self.assertEqual(row[2], "cup/glass/bottle")

    def test_person_is_still_slot_zero(self):
        self.strict()
        _, names = self.load(self.table_leaves())
        for row in names:
            self.assertEqual(row[0], "person")

    def test_the_record_says_the_key_was_the_class_not_a_track_id(self):
        """AG ships no instance id; claiming otherwise would be a fabrication."""
        self.strict()
        mod, _ = self.load(self.table_leaves())
        self.assertEqual(mod.last_load_metadata["slots"]["id_source"], "class")

    def test_the_counts_sit_beside_dropped(self):
        self.strict()
        mod, _ = self.load(self.table_leaves())
        self.assertIn("slots", mod.last_load_metadata)
        self.assertIn("dropped", mod.last_load_metadata)


class TestActionGenomeVacantPersonSlot(ActionGenomeCase):
    """Slot 0 is only person's when person is there."""

    def test_a_frame_without_a_person_leaves_slot_zero_padded(self):
        self.strict()
        ann_dir, frm_dir = self.build_ag(self.table_leaves(), person_frames=[0, 2])
        mod = self.loader()
        _, _, names, _ = mod.build_dataset(annotations_dir=ann_dir,
                                           frames_dir=frm_dir,
                                           num_objs=3, patch_size=4)
        self.assertEqual(names[1][0], "pad_0")
        self.assertEqual(names[1][2], "cup/glass/bottle")


if __name__ == "__main__":
    unittest.main()
