#!/usr/bin/env python3
"""Every dataset must arrive in the same shape, from a registry entry.

The oracle needs boxes and nothing else. That is the fact that makes a second
and a third dataset cost days instead of weeks: on 2026-08-31 eight
Something-Else clips were scored with no video on disk at all. The cost that
remained was in the code, not in the data — each dataset had its own reader with
its own return shape and its own branch through the oracle command line.

So these tests hold two things:

1. **One shape.** Whatever the dataset, a load returns `(boxes, meta)` with
   boxes `(n_frames, num_objs, 4)` in canvas pixels, and `meta` carrying the
   same keys. The loop at the end asserts that over every registered dataset at
   once, which is the only test here that would fail if somebody added a
   dataset that returns something else.
2. **The Action Genome box trap.** Object records are `xywh` and person
   records are `xyxy`, in two files of the same release. Measured over 7,841
   records: 100% of object records are consistent with `xywh` and only 19%
   with `xyxy`. Reading one as the other puts plausible boxes in the wrong
   place and raises nothing, which is the worst failure this project can have.

Every fixture here is written inline into a temporary directory. Nothing reads
the datasets themselves, so the suite stays fast and runs where the data is
absent — which is most machines.
"""

import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from tools.planner import box_loader  # noqa: E402


def _has_pillow():
    """The canvas scaler needs pillow. Bare python3 usually has none."""
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


needs_pillow = unittest.skipUnless(
    _has_pillow(), "pillow is not installed in this interpreter; run under "
                   ".venv-local")


# --- fixtures --------------------------------------------------------------

def write_vidvrd(root, clip_id="ILSVRC2015_train_00005004", frames=6,
                 width=640, height=360, fps=30, motion=5.0):
    """A VidVRD-schema annotation file.

    VidOR uses the identical schema, and `tools/synth_bbox.py` writes it for
    VideoNet with no `fps` and no `relation_instances`, so `fps=None` produces
    what that route produces.
    """
    trajectories = []
    for f in range(frames):
        shift = motion * f
        trajectories.append([
            {"tid": 0, "bbox": {"xmin": 10.0 + shift, "ymin": 20.0,
                                "xmax": 110.0 + shift, "ymax": 140.0}},
            {"tid": 1, "bbox": {"xmin": 300.0, "ymin": 50.0,
                                "xmax": 340.0, "ymax": 90.0}},
        ])
    doc = {"video_id": clip_id.split("/")[-1], "width": width,
           "height": height, "trajectories": trajectories}
    if fps is not None:
        doc["fps"] = fps
    path = os.path.join(root, clip_id + ".json")
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        json.dump(doc, handle)
    return clip_id


def se_label(x, category, y=0.0, size=10.0):
    return {"box2d": {"x1": x, "y1": y, "x2": x + size, "y2": y + size},
            "standard_category": category}


def write_something_else(root, clip_id="13201", frames=4):
    """A Something-Else annotation file: video id to a list of frames."""
    records = []
    for f in range(frames):
        labels = [se_label(10.0 * f, "0000"), se_label(100.0 + 5.0 * f, "hand")]
        if f == frames - 1:                      # the hand leaves the frame
            labels = labels[:1]
        records.append({"name": clip_id + "/%04d.jpg" % (f + 1),
                        "nr_instances": len(labels), "labels": labels})
    path = os.path.join(root, "bounding_box_sample.json")
    with open(path, "w") as handle:
        json.dump({clip_id: records}, handle)
    return clip_id


def write_actiongenome(root, clip_id="001YG.mp4", frames=(89, 92, 95, 98),
                       size=(480, 270), person_mode="xyxy", extra_person=None):
    """The two pickles of one Action Genome release, with one clip in them.

    The keys are `clip/frame.png`, which is the released format; the loader has
    to split them back into a clip and a frame number.
    """
    objects, person = {}, {}
    for i, f in enumerate(frames):
        key = "%s/%06d.png" % (clip_id, f)
        objects[key] = [{"class": "cup", "visible": True,
                         "bbox": [10.0 + i, 20.0, 30.0, 40.0]}]
        person[key] = {
            "bbox": np.array([[100.0 + i, 50.0, 160.0, 200.0]],
                             dtype=np.float32),
            "bbox_size": size,
            "bbox_mode": person_mode,
            "bbox_score": np.array([0.99], dtype=np.float32),
        }
    if extra_person:
        for f in extra_person:
            person["%s/%06d.png" % (clip_id, f)] = {
                "bbox": np.zeros((0, 4), dtype=np.float32),
                "bbox_size": size, "bbox_mode": person_mode}
    with open(os.path.join(root, box_loader.AG_OBJECT_FILE), "wb") as handle:
        pickle.dump(objects, handle)
    with open(os.path.join(root, box_loader.AG_PERSON_FILE), "wb") as handle:
        pickle.dump(person, handle)
    return clip_id


class Fixtures(unittest.TestCase):
    """A temporary root per dataset, torn down after each test."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="box_loader_")
        box_loader.clear_caches()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        box_loader.clear_caches()

    def root(self, name):
        path = os.path.join(self.tmp, name)
        os.makedirs(path)
        return path


# --- the registry ----------------------------------------------------------

class TestRegistry(Fixtures):

    def test_the_corpora_measured_so_far_are_registered(self):
        names = box_loader.datasets()
        for expected in ("actiongenome", "something_else", "vidvrd", "vidor",
                         "videonet"):
            self.assertIn(expected, names)

    def test_an_unknown_corpus_names_the_known_ones(self):
        """Adding a dataset is registering one, so the error must say what is
        registered rather than only that the name is wrong."""
        try:
            box_loader.load_clip("nonesuch", "x")
        except KeyError as exc:
            self.assertIn("something_else", str(exc))
        else:
            self.fail("an unknown dataset must raise")

    def test_every_entry_carries_a_lister_and_a_loader(self):
        for name in box_loader.datasets():
            entry = box_loader.get(name)
            self.assertTrue(callable(entry.list_clips), name)
            self.assertTrue(callable(entry.load_clip), name)
            self.assertTrue(entry.root, name)

    def test_listing_a_corpus_that_is_not_on_disk_is_empty_not_an_error(self):
        """The survey has to run on a machine with no datasets at all."""
        for name in box_loader.datasets():
            self.assertEqual(
                box_loader.list_clips(name, root=os.path.join(self.tmp, "no")),
                [], name)


# --- the Action Genome box conventions -------------------------------------

class TestActionGenomeBoxConventions(Fixtures):
    """Object records are xywh, person records are xyxy. Measured, not guessed."""

    def test_xywh_and_xyxy_do_not_agree(self):
        """The guard is worth nothing if the two readings coincide."""
        self.assertEqual(box_loader.to_xyxy([10, 20, 30, 40], "xywh"),
                         (10.0, 20.0, 40.0, 60.0))
        self.assertEqual(box_loader.to_xyxy([10, 20, 30, 40], "xyxy"),
                         (10.0, 20.0, 30.0, 40.0))

    def test_an_unknown_mode_raises_rather_than_guessing(self):
        self.assertRaises(ValueError, box_loader.to_xyxy, [1, 2, 3, 4], "cxcywh")

    @needs_pillow
    def test_the_object_box_is_read_as_xywh_and_the_person_as_xyxy(self):
        """The one failure that would produce plausible boxes in wrong places.

        The object record `[10, 20, 30, 40]` covers x from 10 to 40 under the
        released convention, and x from 10 to 30 under the wrong one. The
        person record with the same numbers covers x from 10 to 30. So the two
        canvas boxes must differ, and each must match its own convention.
        """
        root = self.root("ag")
        clip = write_actiongenome(root)
        boxes, meta = box_loader.load_clip("actiongenome", clip, root=root,
                                           num_objs=2)
        scale, _, _ = box_loader.canvas_scaler()
        width, height = meta["source_size"]

        cup = meta["slots"].index("cup")
        person = meta["slots"].index("person")
        expect_cup = scale(box_loader.to_xyxy([10.0, 20.0, 30.0, 40.0], "xywh"),
                           width, height)
        expect_person = scale(
            box_loader.to_xyxy([100.0, 50.0, 160.0, 200.0], "xyxy"),
            width, height)
        self.assertEqual(tuple(boxes[0, cup]), tuple(float(v)
                                                     for v in expect_cup))
        self.assertEqual(tuple(boxes[0, person]),
                         tuple(float(v) for v in expect_person))
        # And the wrong reading of the object record is a different box.
        wrong = scale(box_loader.to_xyxy([10.0, 20.0, 30.0, 40.0], "xyxy"),
                      width, height)
        self.assertNotEqual(tuple(expect_cup), tuple(wrong))

    @needs_pillow
    def test_a_release_that_changes_the_person_convention_stops_the_load(self):
        """`bbox_mode` says `xyxy` in the release. If it ever says otherwise the
        boxes would move silently, so the load must refuse instead."""
        root = self.root("ag")
        clip = write_actiongenome(root, person_mode="xywh")
        self.assertRaises(ValueError, box_loader.load_clip,
                          "actiongenome", clip, root=root)

    @needs_pillow
    def test_the_person_is_the_last_slot(self):
        """The actor is in nearly every frame, so it goes last and the object
        slots stay put as num_objs varies."""
        root = self.root("ag")
        clip = write_actiongenome(root)
        _, meta = box_loader.load_clip("actiongenome", clip, root=root,
                                       num_objs=2)
        self.assertEqual(meta["slots"][-1], "person")

    def test_a_clip_needs_both_pickles(self):
        root = self.root("ag")
        write_actiongenome(root)
        with open(os.path.join(root, box_loader.AG_PERSON_FILE), "wb") as h:
            pickle.dump({}, h)
        box_loader.clear_caches()
        self.assertEqual(box_loader.list_clips("actiongenome", root=root), [])

    def test_the_clip_list_comes_from_the_keys_of_both_files(self):
        root = self.root("ag")
        clip = write_actiongenome(root)
        self.assertEqual(box_loader.list_clips("actiongenome", root=root),
                         [clip])


# --- Something-Else --------------------------------------------------------

class TestSomethingElse(Fixtures):

    @needs_pillow
    def test_a_clip_loads_from_an_annotation_file_with_no_video(self):
        root = self.root("se")
        clip = write_something_else(root, frames=4)
        boxes, meta = box_loader.load_clip("something_else", clip, root=root,
                                           num_objs=2)
        self.assertEqual(boxes.shape, (4, 2, 4))
        self.assertEqual(meta["slots"], ["0000", "hand"])
        self.assertEqual(meta["clip_id"], clip)

    @needs_pillow
    def test_an_absent_slot_stays_all_zero(self):
        """All-zero means "not in this frame", not "at the origin" (SPEC V30)."""
        root = self.root("se")
        clip = write_something_else(root, frames=4)
        boxes, meta = box_loader.load_clip("something_else", clip, root=root,
                                           num_objs=2)
        hand = meta["slots"].index("hand")
        self.assertEqual(list(boxes[-1, hand]), [0, 0, 0, 0])
        self.assertEqual(meta["absent"], 1)

    def test_the_clip_list_is_the_video_ids(self):
        root = self.root("se")
        clip = write_something_else(root)
        self.assertEqual(box_loader.list_clips("something_else", root=root),
                         [clip])

    @needs_pillow
    def test_a_single_annotation_file_may_be_given_as_the_root(self):
        """The release is four large files. Naming one avoids opening them all."""
        root = self.root("se")
        clip = write_something_else(root)
        path = os.path.join(root, "bounding_box_sample.json")
        boxes, _ = box_loader.load_clip("something_else", clip, root=path,
                                        num_objs=2)
        self.assertEqual(len(boxes), 4)

    def test_an_unknown_clip_says_which_file_was_searched(self):
        root = self.root("se")
        write_something_else(root)
        try:
            box_loader.load_clip("something_else", "no_such_video", root=root)
        except KeyError as exc:
            self.assertIn("no_such_video", str(exc))
        else:
            self.fail("an unknown clip must raise")


# --- VidVRD and VidOR ------------------------------------------------------

class TestVidvrdFamily(Fixtures):

    @needs_pillow
    def test_vidvrd_loads_through_the_registry(self):
        root = self.root("vidvrd")
        clip = write_vidvrd(root, frames=6)
        boxes, meta = box_loader.load_clip("vidvrd", clip, root=root,
                                           num_objs=2)
        self.assertEqual(boxes.shape, (6, 2, 4))
        self.assertEqual(meta["source_size"], (640, 360))

    @needs_pillow
    def test_vidor_clip_ids_keep_their_folder(self):
        """VidOR stores clips in per-video folders, so the id is `0001/2793...`
        and a bare basename would not find the file."""
        root = self.root("vidor")
        clip = write_vidvrd(root, clip_id="training/0001/2793806282", frames=5)
        listed = box_loader.list_clips("vidor", root=root)
        self.assertEqual(listed, [clip])
        boxes, meta = box_loader.load_clip("vidor", clip, root=root)
        self.assertEqual(len(boxes), 5)
        self.assertEqual(meta["clip_id"], clip)

    @needs_pillow
    def test_videonet_reads_what_synth_bbox_writes(self):
        """The auto-annotation route: no released annotation, so `synth_bbox`
        writes the VidVRD schema with no fps and no relation_instances."""
        root = self.root("videonet")
        clip = write_vidvrd(root, clip_id="A/train/0f1e2d3c", frames=4,
                            fps=None)
        boxes, meta = box_loader.load_clip("videonet", clip, root=root,
                                           num_objs=2)
        self.assertEqual(len(boxes), 4)
        self.assertEqual(meta["dataset"], "videonet")
        self.assertIn("AUTO-ANNOTATED", box_loader.get("videonet").note)

    def test_listing_finds_annotations_in_nested_folders(self):
        root = self.root("vidvrd")
        write_vidvrd(root, clip_id="test/ILSVRC2015_train_00005004")
        write_vidvrd(root, clip_id="train/ILSVRC2015_train_00010001")
        self.assertEqual(len(box_loader.list_clips("vidvrd", root=root)), 2)


# --- the shape every dataset shares -----------------------------------------

class TestOneShape(Fixtures):
    """The point of the unit: a caller writes one code path, not four."""

    def each_corpus(self):
        """(dataset, root, clip_id) for every registered dataset, from fixtures."""
        ag = self.root("ag")
        se = self.root("se")
        vd = self.root("vidvrd")
        vo = self.root("vidor")
        vn = self.root("videonet")
        return [
            ("actiongenome", ag, write_actiongenome(ag)),
            ("something_else", se, write_something_else(se)),
            ("vidvrd", vd, write_vidvrd(vd)),
            ("vidor", vo, write_vidvrd(vo, clip_id="training/0001/2793806282")),
            ("videonet", vn, write_vidvrd(vn, clip_id="A/train/0f1e2d3c",
                                          fps=None)),
        ]

    @needs_pillow
    def test_every_corpus_returns_the_same_keys(self):
        for name, root, clip in self.each_corpus():
            boxes, meta = box_loader.load_clip(name, clip, root=root,
                                               num_objs=3)
            self.assertEqual(sorted(meta), sorted(box_loader.META_KEYS), name)
            self.assertEqual(meta["dataset"], name)
            self.assertEqual(meta["clip_id"], clip)
            self.assertEqual(meta["objects"], 3, name)
            self.assertEqual(meta["frames"], len(boxes), name)
            self.assertEqual(meta["canvas"], box_loader.CANVAS, name)

    @needs_pillow
    def test_every_corpus_returns_canvas_boxes_of_the_same_rank(self):
        for name, root, clip in self.each_corpus():
            boxes, _ = box_loader.load_clip(name, clip, root=root, num_objs=3)
            self.assertEqual(boxes.ndim, 3, name)
            self.assertEqual(boxes.shape[1:], (3, 4), name)
            self.assertEqual(boxes.dtype, np.float32, name)
            self.assertGreater(len(boxes), 0, name)
            width, height = box_loader.CANVAS
            self.assertLessEqual(float(boxes[:, :, 0].max()), width, name)
            self.assertLessEqual(float(boxes[:, :, 1].max()), height, name)

    @needs_pillow
    def test_the_absent_count_is_the_number_of_all_zero_slots(self):
        """One definition of "absent" across datasets, so the survey can add
        them up. The readers count it their own way; the two must agree."""
        for name, root, clip in self.each_corpus():
            boxes, meta = box_loader.load_clip(name, clip, root=root,
                                               num_objs=3)
            zeros = int((np.abs(boxes).sum(axis=-1) == 0).sum())
            self.assertEqual(meta["absent"], zeros, name)

    @needs_pillow
    def test_the_oracle_encodes_what_every_corpus_returns(self):
        """The whole point: boxes in, planner latents out, no branch per dataset."""
        from tools.planner.oracle import boxes_to_latents

        for name, root, clip in self.each_corpus():
            boxes, _ = box_loader.load_clip(name, clip, root=root, num_objs=3)
            latents = boxes_to_latents(boxes, bins_x=8, bins_y=8)
            self.assertEqual(len(latents), len(boxes), name)
            self.assertGreater(int(latents.sum()), 0, name)


# --- the survey ------------------------------------------------------------

class TestSurvey(Fixtures):

    @needs_pillow
    def test_a_survey_row_counts_clips_frames_and_objects(self):
        root = self.root("ag")
        write_actiongenome(root, frames=(89, 92, 95, 98))
        row = box_loader.survey("actiongenome", root=root, limit=10,
                                num_objs=4)
        self.assertTrue(row["available"])
        self.assertEqual(row["clips_listed"], 1)
        self.assertEqual(row["clips_loaded"], 1)
        self.assertEqual(row["median_frames"], 4)
        self.assertEqual(row["median_objects"], 2)   # one cup, one person

    @needs_pillow
    def test_a_clip_that_never_moves_is_counted_as_dead(self):
        """A clip can be long, full of objects and still worth nothing.

        If nothing moves by a whole bin, every frame encodes to the same
        latent, the action set is empty and there is no plan to find. Four
        exports were found on 2026-09-05 carrying one latent with every bit
        zero, so this is a real failure and not a hypothetical one.
        """
        root = self.root("vidvrd")
        write_vidvrd(root, clip_id="still", frames=20, motion=0.0)
        row = box_loader.survey("vidvrd", root=root, limit=5)
        self.assertEqual(row["clips_loaded"], 1)
        self.assertEqual(row["median_distinct"], 1)
        self.assertEqual(row["dead_clips"], 1)

    @needs_pillow
    def test_a_clip_that_moves_carries_more_than_one_state(self):
        root = self.root("vidvrd")
        write_vidvrd(root, clip_id="moving", frames=20, motion=40.0)
        row = box_loader.survey("vidvrd", root=root, limit=5)
        self.assertGreater(row["median_distinct"], 1)
        self.assertEqual(row["dead_clips"], 0)

    @needs_pillow
    def test_a_loadable_clip_is_not_yet_a_plannable_one(self):
        """Action Genome loads nearly every clip and most hold three states.
        A survey that only counted loads would read as a promise it cannot
        keep, so short clips are counted separately."""
        root = self.root("ag")
        write_actiongenome(root, frames=(89, 92, 95, 98))
        self.assertEqual(box_loader.survey("actiongenome", root=root,
                                           min_frames=8)["long_clips"], 0)
        box_loader.clear_caches()
        self.assertEqual(box_loader.survey("actiongenome", root=root,
                                           min_frames=4)["long_clips"], 1)

    def test_a_corpus_with_no_data_reports_unavailable_rather_than_failing(self):
        row = box_loader.survey("vidvrd", root=os.path.join(self.tmp, "gone"))
        self.assertFalse(row["available"])
        self.assertEqual(row["clips_loaded"], 0)

    @needs_pillow
    def test_a_clip_that_cannot_be_read_is_counted_not_fatal(self):
        """One broken file in 7,000 must not end a survey of the dataset."""
        root = self.root("vidvrd")
        write_vidvrd(root, clip_id="good")
        with open(os.path.join(root, "broken.json"), "w") as handle:
            handle.write("{not json")
        row = box_loader.survey("vidvrd", root=root, limit=10)
        self.assertEqual(row["clips_listed"], 2)
        self.assertEqual(row["clips_loaded"], 1)
        self.assertEqual(row["failed"], 1)


class TestCommandLine(Fixtures):

    def test_a_root_can_be_given_per_corpus(self):
        """One survey has to cover a dataset that sits somewhere unusual — a
        sample file, say — beside the datasets that sit where they belong."""
        self.assertEqual(
            box_loader.parse_roots(["something_else=/data/sample.json"],
                                   box_loader.datasets()),
            {"something_else": "/data/sample.json"})

    def test_a_bare_root_needs_a_single_corpus(self):
        self.assertEqual(box_loader.parse_roots(["/data/x"], ["vidvrd"]),
                         {"vidvrd": "/data/x"})
        self.assertRaises(SystemExit, box_loader.parse_roots,
                          ["/data/x"], ["vidvrd", "vidor"])

    def test_a_root_for_an_unregistered_corpus_is_refused(self):
        self.assertRaises(SystemExit, box_loader.parse_roots,
                          ["nonesuch=/data/x"], box_loader.datasets())

    @needs_pillow
    def test_the_command_writes_a_table_and_a_figure_that_open(self):
        """The unit's output is a view of what is reachable. It has to exist
        and it has to be valid XML, which this project has got wrong three
        times by writing a raw angle bracket into SVG text."""
        root = self.root("vidvrd")
        write_vidvrd(root, clip_id="test/ILSVRC2015_train_00005004")
        out = os.path.join(self.tmp, "out")
        code = box_loader.main(["--dataset", "vidvrd", "--root", root,
                                "--limit", "5", "--out-dir", out])
        self.assertEqual(code, 0)

        with open(os.path.join(out, "box_loader_reach.json")) as handle:
            written = json.load(handle)
        self.assertEqual(written["datasets"][0]["clips_loaded"], 1)

        import xml.dom.minidom
        xml.dom.minidom.parse(os.path.join(out, "box_loader_reach.svg"))


class TestFigure(Fixtures):

    def test_the_figure_escapes_angle_brackets(self):
        """A raw < in SVG text is invalid XML. This project has shipped an
        unopenable figure that way three times."""
        rows = [{"dataset": "vidvrd", "available": True, "clips_listed": 3,
                 "clips_loaded": 3, "long_clips": 3, "failed": 0,
                 "median_frames": 128, "median_objects": 2,
                 "median_distinct": 40, "dead_clips": 0,
                 "root": "data/video/vidvrd", "note": "gap <= 6 and 2 > 1"}]
        svg = box_loader.render_svg(rows)
        self.assertNotIn("<= 6", svg)
        self.assertIn("&lt;= 6", svg)
        import xml.dom.minidom
        xml.dom.minidom.parseString(svg)

    def test_every_header_keeps_its_column_of_values(self):
        """The header row and the value rows come from one list. They were two
        parallel tuples, and shortening one of them raised IndexError from
        inside the figure writer rather than anywhere near the mistake."""
        columns = box_loader._columns(8)
        rows = [{"dataset": "vidvrd", "available": True, "clips_listed": 3,
                 "clips_loaded": 3, "long_clips": 3, "failed": 0,
                 "median_frames": 128, "median_objects": 2,
                 "median_distinct": 40, "dead_clips": 0,
                 "root": "data/video/vidvrd", "note": ""}]
        svg = box_loader.render_svg(rows)
        for _, header, _ in columns:
            self.assertIn(box_loader._esc(header), svg)
        # One header line and one value line per column.
        self.assertEqual(svg.count('y="96"'), len(columns))
        self.assertEqual(svg.count('y="122"'), len(columns))

    def test_a_dead_corpus_says_so_under_the_table(self):
        rows = [{"dataset": "actiongenome", "available": True,
                 "clips_listed": 200, "clips_loaded": 200, "long_clips": 8,
                 "failed": 0, "median_frames": 3, "median_objects": 3,
                 "median_distinct": 1, "dead_clips": 96,
                 "root": box_loader.get("actiongenome").root, "note": "x"}]
        svg = box_loader.render_svg(rows)
        self.assertIn("96 of 200 clips carry one state only", svg)

    def test_a_corpus_read_from_elsewhere_says_so_in_the_figure(self):
        """A number from a 13-video sample must not read as the dataset."""
        rows = [{"dataset": "something_else", "available": True,
                 "clips_listed": 13, "clips_loaded": 13, "failed": 0,
                 "median_frames": 46, "median_objects": 2,
                 "root": "notes/lit/samples/something_else_sample.json",
                 "note": ""}]
        svg = box_loader.render_svg(rows)
        self.assertIn("not from its usual root", svg)
        import xml.dom.minidom
        xml.dom.minidom.parseString(svg)

    def test_an_unavailable_corpus_still_gets_a_row(self):
        """The figure answers "which datasets are reachable", so the ones that
        are not have to appear."""
        rows = [{"dataset": "videonet", "available": False, "clips_listed": 0,
                 "clips_loaded": 0, "failed": 0, "median_frames": None,
                 "median_objects": None, "root": "data/video/videonet",
                 "note": ""}]
        svg = box_loader.render_svg(rows)
        self.assertIn("videonet", svg)


if __name__ == "__main__":
    unittest.main()
