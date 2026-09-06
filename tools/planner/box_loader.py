#!/usr/bin/env python3
"""One box-only reader per dataset, behind one registry and one shape.

**The oracle needs boxes and nothing else.** That was measured on 2026-08-31,
when eight Something-Else clips produced plan-validity numbers with no video
downloaded at all. Frames are needed to *train* FOSAE; they are never needed to
*run* the oracle. It is the single fact that makes a second and a third dataset
cost days instead of weeks, and this thesis rests too many headline numbers on
one dataset.

What stood in the way was not the data, it was the code. `oracle.py` grew one
bespoke reader per dataset, each with a slightly different return shape, and
each new dataset added another flag to the command line and another branch
behind it. This module removes that cost:

- every dataset is a **registry entry**, so adding one is writing a function and
  calling `register`, not editing a chain of `if` statements;
- every load returns the **same pair**, `(boxes, meta)`, with boxes shaped
  `(n_frames, num_objs, 4)` in canvas pixels and `meta` carrying `META_KEYS`;
- the readers themselves are **not rewritten**. Where `oracle.py` already has a
  measured reader, the entry calls it. What is new is the common shape, the
  common entry point and the per-dataset indexing that finds the clips.

    from tools.planner import box_loader
    for clip in box_loader.list_clips("actiongenome", limit=20):
        boxes, meta = box_loader.load_clip("actiongenome", clip)

Run it as a script to survey what is reachable on this machine without any
download, which is the question the unit exists to answer::

    python3 tools/planner/box_loader.py --limit 200

It writes `eval/datasets/box_loader_reach.svg` and a JSON beside it.

The Action Genome box trap
--------------------------

In Action Genome the **object** boxes are stored as `xywh` and the **person**
boxes as `xyxy`, in two files of the same release. Measured over 7,841 records:
100% of object records are consistent with `xywh` and only 19% with `xyxy`.
Reading one as the other produces plausible boxes in the wrong places and
raises nothing, which is the worst failure available to this project. So the
conversion is named here (`to_xyxy`), the person file's own `bbox_mode` field
is checked before any box is read, and a load refuses rather than guesses.

Standard library plus numpy, and pillow by way of the canvas scaler. Python 3.6
clean, because the cluster runs 3.6.1.
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

# Importable as a module and runnable as a script from the repository root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir))

from tools.planner import oracle  # noqa: E402

# The canvas every dataset is scaled onto, as (width, height). Taken from the
# oracle so there is one definition rather than a copy (SPEC V5).
CANVAS = (oracle.CANVAS_W, oracle.CANVAS_H)

# The keys `meta` always carries, whatever the dataset. A caller that reads only
# these needs no knowledge of which dataset it has.
#
#   dataset       the registry name
#   clip_id      the id `list_clips` returned, and `load_clip` accepts
#   frames       states in the clip, which is len(boxes)
#   objects      slots per state, which is num_objs
#   slots        what each column holds, when the reader names them
#   absent       slot-frames left all-zero, meaning "not in this frame"
#   source_size  the video's own (width, height), or None when unrecorded
#   canvas       the (width, height) the boxes are scaled to
META_KEYS = ("dataset", "clip_id", "frames", "objects", "slots", "absent",
             "source_size", "canvas")

OUT_DIR = os.path.join("eval", "datasets")

# Action Genome ships two pickles, and both are needed for any clip.
AG_OBJECT_FILE = "object_bbox_and_relationship.pkl"
AG_PERSON_FILE = "person_bbox.pkl"

# Something-Something v2 frames are 240 pixels high with a variable width; the
# annotation records carry no size, so the release's documented 427x240 is
# used. `oracle.boxes_from_something_else` holds the same default.
SOMETHING_ELSE_SIZE = (427, 240)


def canvas_scaler():
    """`(scale, canvas_w, canvas_h)`, the one definition of canvas geometry.

    Re-exported from the oracle so every dataset in this file scales the same
    way and no module here copies the arithmetic.
    """
    return oracle.load_canvas_scaler()


# --- box conventions -------------------------------------------------------

def to_xyxy(box, mode):
    """A four-number box as `(x1, y1, x2, y2)`, whatever the source wrote.

    `mode` is `xyxy` for corner pairs and `xywh` for an origin plus a size.
    An unknown mode raises: a dataset whose convention is not known must stop
    the load, because the two readings differ by a whole box and nothing
    downstream can tell them apart.

    Action Genome needs both modes in one clip. See the module docstring.
    """
    a, b, c, d = [float(v) for v in box]
    if mode == "xyxy":
        return (a, b, c, d)
    if mode == "xywh":
        return (a, b, a + c, b + d)
    raise ValueError("unknown box mode %r; use xyxy or xywh" % (mode,))


# --- the registry ----------------------------------------------------------

class Dataset(object):
    """One annotated dataset the oracle can read from boxes alone.

    `list_clips(root, limit)` returns clip ids; `load_clip(root, clip_id,
    num_objs, **kwargs)` returns `(boxes, meta)`. Neither is called directly by
    a caller: `box_loader.list_clips` and `box_loader.load_clip` add the
    default root and the common shape.
    """

    def __init__(self, name, root, list_clips, load_clip, note=""):
        self.name = name
        self.root = root
        self.list_clips = list_clips
        self.load_clip = load_clip
        self.note = note


REGISTRY = {}
_ORDER = []


def register(dataset):
    """Add a dataset. This is the whole cost of supporting a new one."""
    if dataset.name not in REGISTRY:
        _ORDER.append(dataset.name)
    REGISTRY[dataset.name] = dataset
    return dataset


def datasets():
    """Registered names, in registration order."""
    return list(_ORDER)


def get(name):
    if name not in REGISTRY:
        raise KeyError("unknown dataset %r; registered: %s"
                       % (name, ", ".join(datasets())))
    return REGISTRY[name]


def _finish(name, clip_id, boxes, meta):
    """Force one reader's output into the shape every dataset shares.

    `absent` is recomputed here from the boxes rather than taken from the
    reader, so the number means the same thing in every dataset and a survey can
    add them up. Each reader already counts absent slot-frames its own way, and
    the two agree: an absent slot is written as an all-zero box.
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError("%s/%s: boxes must be (frames, objects, 4); got %s"
                         % (name, clip_id, (boxes.shape,)))
    source = (meta or {}).get("source_size")
    out = {
        "dataset": name,
        "clip_id": clip_id,
        "frames": int(boxes.shape[0]),
        "objects": int(boxes.shape[1]),
        "slots": list((meta or {}).get("slots") or []),
        "absent": int((np.abs(boxes).sum(axis=-1) == 0).sum()),
        "source_size": tuple(source) if source else None,
        "canvas": tuple((meta or {}).get("canvas") or CANVAS),
    }
    return boxes, out


def list_clips(name, root=None, limit=None):
    """Clip ids for a dataset, in a stable order.

    A root that does not exist gives an empty list rather than an error: the
    survey has to run on a machine that holds none of the datasets, and that is
    most machines.
    """
    entry = get(name)
    return entry.list_clips(root or entry.root, limit)


def load_clip(name, clip_id, num_objs=3, root=None, **kwargs):
    """`(boxes, meta)` for one clip of one dataset. Boxes are canvas pixels."""
    entry = get(name)
    boxes, meta = entry.load_clip(root or entry.root, clip_id, num_objs,
                                  **kwargs)
    return _finish(name, clip_id, boxes, meta)


def clear_caches():
    """Drop the parsed annotation files held between loads."""
    _AG_CACHE.clear()
    _SE_CACHE["path"] = None
    _SE_CACHE["data"] = None


# --- VidVRD, and VidOR, which uses the identical schema ---------------------

def _json_clips(root, limit=None):
    """Clip ids for a tree of annotation JSON files: the path, minus `.json`.

    VidOR keeps clips in per-video folders, so the id has to keep the folder.
    A bare basename would not find the file and would collide across folders.
    """
    if not os.path.isdir(root):
        return []
    found = []
    for path in glob.glob(os.path.join(root, "*.json")):
        found.append(os.path.basename(path)[:-len(".json")])
    for path in glob.glob(os.path.join(root, "*", "*.json")):
        found.append(os.path.relpath(path, root)[:-len(".json")])
    for path in glob.glob(os.path.join(root, "*", "*", "*.json")):
        found.append(os.path.relpath(path, root)[:-len(".json")])
    found.sort()
    return found[:limit] if limit else found


def _load_json_clip(root, clip_id, num_objs=3, fill=True):
    """One VidVRD-schema clip, through the oracle's measured reader."""
    path = os.path.join(root, clip_id + ".json")
    if not os.path.isfile(path):
        raise KeyError("clip %r is not under %s" % (clip_id, root))
    return oracle.boxes_from_vidvrd(path, num_objs=num_objs, fill=fill)


register(Dataset(
    "vidvrd", os.path.join("data", "video", "vidvrd", "annotations"),
    _json_clips, _load_json_clip,
    note="one JSON per clip; slots are the tids of the largest tracks"))

register(Dataset(
    "vidor", os.path.join("data", "video", "vidor", "annotations"),
    _json_clips, _load_json_clip,
    note="identical schema to VidVRD, in per-video folders"))


# --- Something-Else --------------------------------------------------------

# One parsed annotation file at a time. The release is four files of several
# hundred megabytes, so holding them all would cost more memory than the rest
# of the pipeline; holding none would re-parse a file per clip, which is what
# made a batch over the dataset slow.
_SE_CACHE = {"path": None, "data": None}


def _se_files(root):
    """The annotation files under a root, or the root itself when it is one."""
    if os.path.isfile(root):
        return [root]
    if not os.path.isdir(root):
        return []
    return sorted(glob.glob(os.path.join(root, "*.json")))


def _se_read(path):
    if _SE_CACHE["path"] != path:
        with open(path) as handle:
            _SE_CACHE["data"] = json.load(handle)
        _SE_CACHE["path"] = path
    return _SE_CACHE["data"]


def _se_clips(root, limit=None):
    found = []
    for path in _se_files(root):
        try:
            found.extend(_se_read(path))
        except ValueError:                   # a file that is not an annotation
            continue
        if limit and len(found) >= limit:
            break
    found.sort()
    return found[:limit] if limit else found


def _se_load(root, clip_id, num_objs=3, width=None, height=None):
    """One Something-Else video, through the oracle's measured reader.

    The video is looked up across the files under `root`, so a caller needs the
    video id and not the part number. Naming one file as the root skips that
    search, which matters when the other parts are hundreds of megabytes.
    """
    files = _se_files(root)
    for path in files:
        try:
            data = _se_read(path)
        except ValueError:
            continue
        if clip_id in data:
            size = (width or SOMETHING_ELSE_SIZE[0],
                    height or SOMETHING_ELSE_SIZE[1])
            boxes, meta = oracle.boxes_from_something_else_frames(
                data[clip_id], size[0], size[1], num_objs=num_objs)
            meta["source_size"] = size
            return boxes, meta
    raise KeyError("video %r is in none of the %d annotation files under %s"
                   % (clip_id, len(files), root))


# VideoNet has no released annotation. `tools/synth_bbox.py` writes the VidVRD
# schema for it from MediaPipe hands and Grounded-DINO, so the same reader
# serves it — which is the whole reason that tool writes that schema.
register(Dataset(
    "videonet", os.path.join("data", "video", "videonet", "annotations"),
    _json_clips, _load_json_clip,
    note="AUTO-ANNOTATED by tools/synth_bbox.py, not ground truth"))


register(Dataset(
    "something_else", os.path.join("data", "video", "something_else", "raw"),
    _se_clips, _se_load,
    note="per-frame boxes for 180,049 videos; no frames needed for the oracle"))


# --- Action Genome ---------------------------------------------------------

# The two pickles cost a few hundred megabytes and a minute to parse, so a
# batch over the dataset must parse them once. Keyed on the root.
_AG_CACHE = {}


def _ag_group(raw):
    """`{clip: {frame_number: value}}` from keys shaped `clip/000089.png`.

    `tools/video/screen_actiongenome.load_by_clip` does the same regrouping for
    the object file alone. It stays there because the screen reads the dataset
    for a different purpose; both files need it here.
    """
    by_clip = {}
    for key, value in raw.items():
        clip, frame = key.split("/")
        by_clip.setdefault(clip, {})[int(frame.split(".")[0])] = value
    return by_clip


def ag_index(root):
    """`(objects_by_clip, person_by_clip)`, parsed once per root.

    The annotation files are pickles because that is the format Action Genome
    publishes; the release ships no JSON alternative. They are the official
    download, so they are trusted input. Nothing here unpickles anything that
    arrives from elsewhere.
    """
    root = os.path.abspath(root)
    if root in _AG_CACHE:
        return _AG_CACHE[root]
    objects_path = os.path.join(root, AG_OBJECT_FILE)
    person_path = os.path.join(root, AG_PERSON_FILE)
    if not (os.path.isfile(objects_path) and os.path.isfile(person_path)):
        _AG_CACHE[root] = ({}, {})
        return _AG_CACHE[root]
    with open(objects_path, "rb") as handle:
        objects = _ag_group(pickle.load(handle))
    with open(person_path, "rb") as handle:
        person = _ag_group(pickle.load(handle))
    _AG_CACHE[root] = (objects, person)
    return _AG_CACHE[root]


def _ag_clips(root, limit=None):
    """Clips present in **both** files. A clip in one alone has no frames."""
    objects, person = ag_index(root)
    found = sorted(set(objects) & set(person))
    return found[:limit] if limit else found


def _ag_check_modes(clip_id, person_by_frame):
    """Refuse the clip if the person file stops saying `xyxy`.

    The person records carry their own `bbox_mode`, and every one of them says
    `xyxy` in the release on disk. The object records carry no such field and
    are `xywh`. If a future release flips either, every box moves and no
    downstream number looks wrong, so the check is worth its two lines.
    """
    modes = set()
    for record in person_by_frame.values():
        mode = record.get("bbox_mode")
        if mode is not None:
            modes.add(mode)
    if modes - set(["xyxy"]):
        raise ValueError(
            "%s: the Action Genome person file declares bbox_mode %s, but the "
            "reader converts it as xyxy. Objects are xywh and persons are "
            "xyxy in the release measured on 7,841 records; a change here "
            "moves every person box." % (clip_id, sorted(modes)))


def _ag_load(root, clip_id, num_objs=3, max_gap=6):
    """One Action Genome clip's densest run, through the oracle's reader.

    Action Genome annotates a fraction of the frames, so only the longest run
    of frames within `max_gap` reads as a sequence of states. The reader picks
    that run; this function finds the clip and guards the box conventions.
    """
    objects, person = ag_index(root)
    if clip_id not in objects or clip_id not in person:
        raise KeyError("clip %r is not in both annotation files under %s"
                       % (clip_id, root))
    _ag_check_modes(clip_id, person[clip_id])
    return oracle.boxes_from_actiongenome_clip(
        objects[clip_id], person[clip_id], num_objs=num_objs, max_gap=max_gap)


register(Dataset(
    "actiongenome", os.path.join("data", "video", "actiongenome",
                                 "annotations"),
    _ag_clips, _ag_load,
    note="objects are xywh and persons are xyxy; the loader normalises both"))


# --- what is reachable without a download ----------------------------------

def _median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def distinct_states(boxes, bins_x=None, bins_y=None):
    """How many **different** latents a clip produces once quantised.

    A dataset can hand over long clips full of objects and still give a planner
    nothing: if the boxes move less than one bin, every frame encodes to the
    same latent, there is no transition to mine and no plan to find. On
    2026-09-05 four exports under `eval/exports` were found holding exactly one
    distinct latent, every bit zero, and no summary anywhere would have shown
    it. So the survey counts this next to the frame count.

    The bins default to the real decoder's resolution, which is what an export
    uses, so the number is the one a planner would actually see.
    """
    latents = oracle.boxes_to_latents(
        boxes,
        bins_x if bins_x else oracle.DEFAULT_BINS_X,
        bins_y if bins_y else oracle.DEFAULT_BINS_Y)
    if not len(latents):
        return 0
    return int(len(np.unique(latents, axis=0)))


def survey(name, root=None, limit=200, num_objs=8, min_frames=8):
    """Count what one dataset yields: clips loadable, frames and objects.

    Reads the **first** `limit` clips in id order rather than a random sample,
    so the numbers are reproducible from the command line alone.

    `median_objects` counts the slots a clip actually fills, up to `num_objs`.
    It is the number that says whether a dataset has anything to relate: a clip
    with one object cannot carry a relation.

    `long_clips` counts the clips reaching `min_frames` states. Loading a clip
    is not the same as being able to plan over it — Action Genome loads almost
    every clip and most of them hold three states — so the two counts are
    reported side by side. The default of 8 is the window
    `screen_actiongenome.py` screens at.

    A clip that will not read is counted in `failed` and the survey continues.
    The readers signal a useless clip by raising `SystemExit`, which a survey
    must not honour, so that is caught too.
    """
    entry = get(name)
    root = root or entry.root
    ids = list_clips(name, root=root, limit=limit)
    frames, objects, distinct, failed = [], [], [], 0
    for clip_id in ids:
        try:
            boxes, _ = load_clip(name, clip_id, num_objs=num_objs, root=root)
        except (Exception, SystemExit):
            failed += 1
            continue
        if not len(boxes):
            failed += 1
            continue
        frames.append(int(boxes.shape[0]))
        occupied = (np.abs(boxes).sum(axis=-1) > 0).any(axis=0)
        objects.append(int(occupied.sum()))
        distinct.append(distinct_states(boxes))
    return {
        "dataset": name,
        "root": root,
        "available": bool(ids),
        "clips_listed": len(ids),
        "clips_loaded": len(frames),
        "long_clips": sum(1 for n in frames if n >= min_frames),
        "min_frames": min_frames,
        "failed": failed,
        "median_frames": _median(frames),
        "median_objects": _median(objects),
        "median_distinct": _median(distinct),
        "dead_clips": sum(1 for n in distinct if n <= 1),
        "num_objs": num_objs,
        "note": entry.note,
    }


def parse_roots(values, names):
    """`--root` arguments as `{dataset: path}`.

    Each value is `name=path`, so one command can survey a dataset that sits
    somewhere other than its registered root — a sample file, or a scratch
    copy — alongside the datasets that sit where they belong. A bare path is
    accepted when exactly one dataset is being surveyed, because that is the
    common case and the pair would only repeat the name.
    """
    roots = {}
    for value in values or []:
        if "=" in value:
            name, path = value.split("=", 1)
            if name not in REGISTRY:
                raise SystemExit("--root names %r, which is not registered: %s"
                                 % (name, ", ".join(datasets())))
            roots[name] = path
        elif len(names) == 1:
            roots[names[0]] = value
        else:
            raise SystemExit("--root %s needs a dataset: write name=path"
                             % (value,))
    return roots


def _esc(text):
    """XML-escape text for an SVG. A raw `<` makes the figure unopenable."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _cell(x, y, text, css="v", anchor="start"):
    return ('<text x="%d" y="%d" class="%s" text-anchor="%s">%s</text>'
            % (x, y, css, anchor, _esc(text)))


def _columns(min_frames):
    """`(x, header, key)` per column, so a header can never lose its value.

    The header row and the value rows are drawn from this one list. They were
    two parallel tuples once, and editing one of them left the other shorter,
    which raised `IndexError` from inside the figure writer rather than
    anywhere near the mistake.
    """
    return (
        (20, "dataset", "dataset"),
        (140, "clips listed", "clips_listed"),
        (232, "clips loaded", "clips_loaded"),
        (330, "clips >= %d frames" % min_frames, "long_clips"),
        (452, "frames/clip", "median_frames"),
        (556, "objects/clip", "median_objects"),
        (652, "states/clip", "median_distinct"),
    )


def render_svg(rows, limit=None):
    """A table: per dataset, the clips loadable and what one clip holds.

    A table rather than a chart because the numbers are not comparable to each
    other and a reader wants to look one of them up, not compare bar heights.
    Datasets that are absent keep a row: "which datasets are reachable without a
    download" is the question, so an empty answer is an answer.

    `states/clip` is the count of **distinct** latents a clip produces once the
    boxes are quantised. It is here because four exports under `eval/exports`
    were found on 2026-09-05 carrying exactly one distinct latent, every bit
    zero, and nothing in any summary would have shown it. A clip whose whole
    trajectory quantises to one state offers the planner no transition at all.
    """
    width = 760
    top = 96
    row_h = 26
    moved = [r for r in rows
             if r["available"] and r["root"] != get(r["dataset"]).root]
    min_frames = ([r.get("min_frames") for r in rows if r.get("min_frames")]
                  or [8])[0]
    num_objs = ([r.get("num_objs") for r in rows if r.get("num_objs")]
                or [8])[0]
    columns = _columns(min_frames)

    # The notes go under the table rather than in a column: truncated to a
    # column width they were unreadable, and each one records why a dataset
    # behaves as it does.
    footnotes = [
        "frames/clip, objects/clip and states/clip are medians over the clips "
        "that loaded, with up to %d slots." % num_objs,
        "Loading a clip is not planning over it: a clip needs distinct states "
        "to plan through, and more than one",
        "object to carry a relation. states/clip counts distinct latents after "
        "quantisation, so a clip at 1 is dead.",
    ]
    for row in rows:
        if not row["available"]:
            footnotes.append("%s -- not on this machine: %s"
                             % (row["dataset"], row["root"]))
            continue
        note = "%s -- %s" % (row["dataset"], row["note"])
        dead = row.get("dead_clips")
        if dead:
            note += " (%d of %d clips carry one state only)" % (
                dead, row["clips_loaded"])
        footnotes.append(note)
    for row in moved:
        footnotes.append("%s was read from %s, not from its usual root."
                         % (row["dataset"], row["root"]))
    height = top + row_h * (len(rows) + 1) + 20 + 14 * len(footnotes)

    left = columns[0][0]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
        'viewBox="0 0 %d %d">' % (width, height, width, height),
        '<style>text{font-family:sans-serif}.t{font-size:15px;'
        'font-weight:bold}.h{font-size:11px;fill:#555;font-weight:bold}'
        '.v{font-size:12px}.n{font-size:10px;fill:#666}'
        '.off{font-size:12px;fill:#999}</style>',
        '<rect width="%d" height="%d" fill="white"/>' % (width, height),
        _cell(left, 30, "Datasets reachable from annotations alone", "t"),
        _cell(left, 50, "The oracle reads boxes. Frames are needed only to "
                        "train, never to run it.", "n"),
        _cell(left, 66, "First %s clips per dataset in id order, not a random "
                        "sample." % (limit if limit else "N"), "n"),
    ]
    for x, header, _ in columns:
        parts.append(_cell(x, top, header, "h"))
    parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#ccc"/>'
                 % (left, top + 8, width - 20, top + 8))

    y = top + row_h
    for row in rows:
        available = row["available"]
        css = "v" if available else "off"
        for x, _, key in columns:
            if key == "dataset":
                value = row["dataset"]
            elif not available:
                value = "-"
            else:
                value = row.get(key)
                value = "-" if value is None else value
            parts.append(_cell(x, y, value, css))
        y += row_h

    for i, line in enumerate(footnotes):
        parts.append(_cell(left, y + 20 + 14 * i, line, "n"))
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Survey which annotated datasets the oracle can read on "
                    "this machine, with no video downloaded.")
    ap.add_argument("--dataset", default="all",
                    help="one registered dataset, or all. Registered: "
                         + ", ".join(datasets()))
    ap.add_argument("--root", action="append", default=None,
                    help="read a dataset from somewhere else, as name=path. "
                         "Repeatable. A bare path is allowed when --dataset "
                         "names one dataset.")
    ap.add_argument("--limit", type=int, default=200,
                    help="clips to read per dataset, in id order")
    ap.add_argument("--num-objs", type=int, default=8,
                    help="slots per state while surveying. Larger than the "
                         "oracle's 3, so objects/clip measures the dataset "
                         "rather than the export setting.")
    ap.add_argument("--min-frames", type=int, default=8,
                    help="states a clip needs to count as long enough to plan "
                         "over. 8 is the window screen_actiongenome.py uses.")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)

    names = datasets() if args.dataset == "all" else [args.dataset]
    roots = parse_roots(args.root, names)

    rows = []
    for name in names:
        row = survey(name, root=roots.get(name), limit=args.limit,
                     num_objs=args.num_objs, min_frames=args.min_frames)
        rows.append(row)
        print("%-15s listed %-6s loaded %-6s long %-6s frames %-7s objects "
              "%-5s states %-7s dead %-5s %s"
              % (row["dataset"], row["clips_listed"], row["clips_loaded"],
                 row["long_clips"], row["median_frames"],
                 row["median_objects"], row["median_distinct"],
                 row["dead_clips"],
                 "" if row["available"] else "(absent: " + row["root"] + ")"))
        clear_caches()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    out_json = os.path.join(args.out_dir, "box_loader_reach.json")
    with open(out_json, "w") as handle:
        json.dump({"limit": args.limit, "num_objs": args.num_objs,
                   "min_frames": args.min_frames, "datasets": rows},
                  handle, indent=2)
    out_svg = os.path.join(args.out_dir, "box_loader_reach.svg")
    with open(out_svg, "w") as handle:
        handle.write(render_svg(rows, limit=args.limit))
    print("wrote %s" % out_json)
    print("wrote %s" % out_svg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
