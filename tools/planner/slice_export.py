#!/usr/bin/env python3
"""Cut one clip out of a multi-clip export, so windows stop crossing videos.

**The defect this exists for.** `export_latents.py` writes every frame of every
clip into one flat sequence. H14's export is 8,610 frames from **88 clips**,
with **87 boundaries** inside it. Every windowing tool in this project slides
blindly along that axis, so a window straddling a boundary pairs the last frame
of one video with the first frame of another — a transition that never
happened, scored as if it had.

At window 8 that is 609 of 8,602 possible starts, about 7%. It is not a
majority, but each one is pure noise entering a median, and the fix is cheap:
the export already carries `frame_ids` of the form ``<video_id>/<frame>``, so
the boundaries are recoverable exactly.

    python3 tools/planner/slice_export.py IN.npz --list
    python3 tools/planner/slice_export.py IN.npz --clip ILSVRC2015_train_00150010 \\
        --out eval/exports/H14-150010.npz
    python3 tools/planner/slice_export.py IN.npz --all --out-dir eval/exports/h14

Every array with one row per frame is sliced together, so `latents`,
`gt_boxes` and `decoded_boxes` cannot drift apart — the failure `SPEC.md` V28
records. Scalars are copied through.

numpy and the standard library only, so it runs on Sherlock's Python 3.6.
"""

import argparse
import os
import sys

import numpy as np


def clip_of(frame_id):
    """`ILSVRC2015_train_00150010/000042` -> `ILSVRC2015_train_00150010`."""
    return str(frame_id).rsplit("/", 1)[0]


def clip_index(frame_ids):
    """Ordered {video_id: (start, stop)} half-open index ranges.

    Contiguity is assumed and checked: a clip whose frames are not consecutive
    in the export would silently produce a slice spanning other clips.
    """
    clips, order = {}, []
    for i, f in enumerate(frame_ids):
        c = clip_of(f)
        if c not in clips:
            clips[c] = [i, i + 1]
            order.append(c)
        else:
            if clips[c][1] != i:
                raise ValueError(
                    "clip %s is not contiguous in the export: frames stop at "
                    "%d and resume at %d" % (c, clips[c][1], i))
            clips[c][1] = i + 1
    return [(c, tuple(clips[c])) for c in order]


def slice_export(data, start, stop):
    """Slice every per-frame array together; copy scalars through."""
    n = len(data["frame_ids"])
    out = {}
    for k in data.files:
        a = data[k]
        if a.ndim >= 1 and a.shape[0] == n:
            out[k] = a[start:stop]
        else:
            out[k] = a
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("export")
    ap.add_argument("--clip", help="video id to cut out")
    ap.add_argument("--out", help="where to write the single-clip export")
    ap.add_argument("--all", action="store_true", help="write every clip")
    ap.add_argument("--out-dir", help="directory for --all")
    ap.add_argument("--min-frames", type=int, default=10,
                    help="skip clips shorter than this under --all")
    ap.add_argument("--list", action="store_true", help="just show the clips")
    a = ap.parse_args(argv)

    # No allow_pickle: every field here is a plain numeric or
    # unicode array, so nothing needs object deserialisation.
    d = np.load(a.export)
    if "frame_ids" not in d.files:
        raise SystemExit(
            "%s has no `frame_ids`, so its clip boundaries are unrecoverable. "
            "Re-export it with a current export_latents.py." % a.export)

    index = clip_index(d["frame_ids"])

    if a.list or not (a.clip or a.all):
        print("%d clips, %d frames" % (len(index), len(d["frame_ids"])))
        for c, (s, e) in index:
            print("  %-34s %5d frames  [%5d:%5d]" % (c, e - s, s, e))
        return 0

    lookup = dict(index)

    if a.clip:
        if a.clip not in lookup:
            raise SystemExit("%s is not in this export" % a.clip)
        if not a.out:
            raise SystemExit("--clip needs --out")
        s, e = lookup[a.clip]
        np.savez_compressed(a.out, **slice_export(d, s, e))
        print("wrote %s  (%s, %d frames)" % (a.out, a.clip, e - s))
        return 0

    if not a.out_dir:
        raise SystemExit("--all needs --out-dir")
    if not os.path.isdir(a.out_dir):
        os.makedirs(a.out_dir)
    written = skipped = 0
    for c, (s, e) in index:
        if e - s < a.min_frames:
            skipped += 1
            continue
        np.savez_compressed(os.path.join(a.out_dir, c + ".npz"),
                            **slice_export(d, s, e))
        written += 1
    print("wrote %d clips to %s, skipped %d shorter than %d frames"
          % (written, a.out_dir, skipped, a.min_frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())
