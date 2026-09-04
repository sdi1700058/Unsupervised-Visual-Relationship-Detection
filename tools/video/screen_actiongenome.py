#!/usr/bin/env python3
"""Screen Action Genome for clips that can produce planner transitions.

Action Genome was set aside in an earlier pass on annotation density, quoting
its paper: *"We do not annotate every single frame in a video"*, and for each
action they *"uniformly sample 5 frames to annotate across the range"*. That is
accurate about the paper and it was the wrong thing to decide on, because
density is a property of a **clip**, not of a corpus. Some clips are annotated
densely and the rest can be ignored.

So this measures rather than assumes. What it reports, per clip:

- the annotated frame numbers, and the **gaps** between them;
- the longest **run** of frames whose gaps stay within a tolerance, which is
  what a sliding window needs;
- objects and non-trivial relations per frame.

A relation is non-trivial when it is not `[]`, not `['unsure']` and not
`['not_contacting']`. Counting those as relations inflates the density of every
frame in the corpus, because the annotation format records them explicitly
rather than by omission.

    python3 tools/video/screen_actiongenome.py
    python3 tools/video/screen_actiongenome.py --max-gap 6 --min-run 8

Writes `eval/datasets/actiongenome_screen.json` and a figure beside it. Reads
one 135 MB pickle, so it holds a few hundred megabytes for a minute; run it
under a memory cap.

Standard library only, Python 3.6 clean.
"""

import argparse
import json
import os
import pickle
import statistics as st


ANNOTATIONS = "data/video/actiongenome/annotations"
OUT_DIR = "eval/datasets"

# The annotation format records "no relation" explicitly, so these are absences
# written down rather than relations.
TRIVIAL = ("[]", "['unsure']", "['not_contacting']", "")

RELATION_FIELDS = ("attention_relationship", "spatial_relationship",
                   "contacting_relationship")


def non_trivial_relations(record):
    """How many real relations one object record carries."""
    total = 0
    for field in RELATION_FIELDS:
        value = record.get(field)
        if value and str(value).strip() not in TRIVIAL:
            total += 1
    return total


def runs_within(frames, max_gap):
    """Lengths of maximal runs whose consecutive gaps are all <= max_gap."""
    if not frames:
        return []
    frames = sorted(frames)
    out, run = [], 1
    for i in range(len(frames) - 1):
        if frames[i + 1] - frames[i] <= max_gap:
            run += 1
        else:
            out.append(run)
            run = 1
    out.append(run)
    return out


def screen_clip(frames, records_by_frame, max_gap):
    """One clip's numbers."""
    frames = sorted(frames)
    gaps = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
    run_lengths = runs_within(frames, max_gap)
    objs, rels = [], []
    for f in frames:
        recs = [r for r in records_by_frame[f] if r.get("visible")]
        objs.append(len(recs))
        rels.append(sum(non_trivial_relations(r) for r in recs))
    return {
        "n_frames": len(frames),
        "median_gap": st.median(gaps) if gaps else None,
        "longest_run": max(run_lengths) if run_lengths else 0,
        "median_objects": st.median(objs) if objs else 0,
        "median_relations": st.median(rels) if rels else 0,
    }


def load_by_clip(path=None):
    """Annotations grouped by clip, then by frame number.

    The annotation file is a pickle because that is the format Action Genome
    publishes; the project ships no JSON alternative. It is the official release
    downloaded by the author, so it is treated as trusted input. Nothing here
    unpickles anything that arrives from elsewhere.

    Loading it costs a few hundred megabytes for a minute, so it is done **once**
    and the result is passed to `screen`. An earlier version re-read the file for
    every gap tolerance in the sweep, which was six loads for one figure.
    """
    path = path or os.path.join(ANNOTATIONS,
                                "object_bbox_and_relationship.pkl")
    with open(path, "rb") as handle:
        raw = pickle.load(handle)

    by_clip = {}
    for key, recs in raw.items():
        clip, frame = key.split("/")
        number = int(frame.split(".")[0])
        by_clip.setdefault(clip, {})[number] = recs
    return by_clip


def screen(by_clip, max_gap=6, min_run=8):
    """Screen the corpus. Returns a summary and the qualifying clip ids."""
    per_clip = {}
    for clip, records_by_frame in by_clip.items():
        per_clip[clip] = screen_clip(sorted(records_by_frame),
                                     records_by_frame, max_gap)

    qualifying = sorted(c for c, v in per_clip.items()
                        if v["longest_run"] >= min_run)

    def med(field):
        values = [v[field] for v in per_clip.values() if v[field] is not None]
        return st.median(values) if values else None

    return {
        "corpus": "actiongenome",
        "max_gap": max_gap,
        "min_run": min_run,
        "clips": len(per_clip),
        "qualifying_clips": len(qualifying),
        "median_frames_per_clip": med("n_frames"),
        "median_gap": med("median_gap"),
        "median_objects_per_frame": med("median_objects"),
        "median_relations_per_frame": med("median_relations"),
        "median_longest_run": med("longest_run"),
        "qualifying": qualifying,
    }


def per_object_tracks(objects_by_frame, person_by_frame, frames, num_objs=3):
    """`{slot: {step: box}}` in source pixels, for the crossover criterion.

    **Steps are consecutive annotated frames, reindexed to 0, 1, 2 ...** The
    criterion in `screen_vidvrd.window_crossover` requires contiguous frame
    numbers, and Action Genome's annotated frames are 1 to `max_gap` source
    frames apart. Treating each annotated frame as one state is legitimate — a
    planner works on a sequence of states, not on wall-clock time — but it means
    **one step spans a variable amount of real time**, between 1 and `max_gap`
    frames. That is the same property `SPEC.md` V35 measures as temporal
    compression, here introduced by the corpus rather than by a model, and it
    has to be stated wherever these numbers are quoted.

    Boxes stay in the video's own pixels because that is what the criterion
    expects. Object records are `xywh` and person records are `xyxy`.
    """
    tracks = {}
    for step, f in enumerate(frames):
        for record in objects_by_frame.get(f, []):
            if not (record.get("visible") and record.get("bbox")):
                continue
            x, y, w, h = record["bbox"]
            tracks.setdefault(record["class"], {})[step] = [x, y, x + w, y + h]
        person = person_by_frame.get(f, {}).get("bbox")
        if person is not None and len(person):
            tracks.setdefault("person", {})[step] = list(person[0])
    # Keep the slots present in most steps, person last, as the loader does.
    ordered = sorted(tracks, key=lambda c: (c == "person", -len(tracks[c]), c))
    return dict((c, tracks[c]) for c in ordered[:num_objs])


def render_svg(summary):
    """A figure: how the usable corpus depends on the gap tolerance."""
    w, h, pad = 640, 300, 56
    bars = summary.get("sweep") or []
    if not bars:
        return None
    top = max(v for _, v in bars) or 1
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
             'viewBox="0 0 %d %d">' % (w, h, w, h),
             '<style>text{font-family:sans-serif}.t{font-size:15px;'
             'font-weight:bold}.l{font-size:11px}.v{font-size:11px;'
             'font-weight:bold}.n{font-size:11px;fill:#555}</style>',
             '<rect width="%d" height="%d" fill="white"/>' % (w, h),
             '<text x="%d" y="28" class="t">Action Genome: clips with a usable '
             'run of %d frames</text>' % (pad, summary["min_run"])]
    step = (w - 2 * pad) // max(len(bars), 1)
    for i, (gap, count) in enumerate(bars):
        height = int(170 * count / top)
        x = pad + i * step
        parts.append('<rect x="%d" y="%d" width="%d" height="%d" '
                     'fill="#2b6cb0"/>' % (x, 215 - height, step - 10, height))
        parts.append('<text x="%d" y="%d" class="v">%d</text>'
                     % (x, 210 - height, count))
        parts.append('<text x="%d" y="234" class="l">gap &#8804;%d</text>'
                     % (x, gap))
    parts.append('<text x="%d" y="266" class="n">%d clips in the corpus. '
                 'VidVRD, for comparison, yields 88 screened clips.</text>'
                 % (pad, summary["clips"]))
    parts.append('<text x="%d" y="282" class="n">A gap is measured in source '
                 'frames; the median gap across the corpus is %s.</text>'
                 % (pad, summary["median_gap"]))
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-gap", type=int, default=6)
    ap.add_argument("--min-run", type=int, default=8)
    ap.add_argument("--sweep", default="2,4,6,8,12",
                    help="gap tolerances to report, for the figure")
    a = ap.parse_args(argv)

    by_clip = load_by_clip()
    summary = screen(by_clip, max_gap=a.max_gap, min_run=a.min_run)

    # The headline number depends on the tolerance, so report the curve rather
    # than one point. A single figure invites treating it as intrinsic.
    sweep = []
    for gap in [int(g) for g in a.sweep.split(",")]:
        sweep.append((gap, screen(by_clip, max_gap=gap,
                                  min_run=a.min_run)["qualifying_clips"]))
    summary["sweep"] = sweep

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    out = os.path.join(OUT_DIR, "actiongenome_screen.json")
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=2)
    print("clips %d, qualifying %d at gap <=%d and run >=%d"
          % (summary["clips"], summary["qualifying_clips"], a.max_gap,
             a.min_run))
    print("median gap %s, objects %s, relations %s per frame"
          % (summary["median_gap"], summary["median_objects_per_frame"],
             summary["median_relations_per_frame"]))
    print("sweep (gap, qualifying): %s" % sweep)
    print("wrote %s" % out)

    svg = render_svg(summary)
    if svg:
        figure = os.path.join(OUT_DIR, "actiongenome_screen.svg")
        with open(figure, "w") as handle:
            handle.write(svg)
        print("wrote %s" % figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
