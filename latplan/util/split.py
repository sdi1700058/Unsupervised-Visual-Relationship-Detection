"""Train, val and test by whole clip, chosen with `SPLIT=clip`.

**The defect this closes.** `strips.py vidvrd()` cut its states 90/5/5 after
`transitions.reshape(...)`, which puts every "before" state first and every
"after" state second. Val therefore held the "after" frames of the last clips,
and frame t+1 of one pair is frame t of the next pair in the same clip, so
nearly every val state was also a training state. A val curve made that way
says nothing about frames the model has not seen. Found 2026-10-03.

**What this does instead.** Clips keep their load order. The last 5 per cent
go to test, the 5 per cent before them to val, and the rest to training, each
at least one clip. Each split then holds the "before" states of its pairs
followed by their "after" states, the same layout the old split used, so the
number of states per pair and the batch make-up stay comparable.

**What it does not do.** It does not choose the pairs. The caller passes the
pairs `latplan.util.adjacency.sequential_pairs` made, so this split and
`build_transitions` agree on which frames are one step apart, including under
`STRICT_ADJACENCY=1`.

This module imports nothing from `latplan`, so it loads without TensorFlow.
"""

import collections

import numpy as np

SPLITS = ("train", "val", "test")


def _video_of(frame_id):
    """The clip of a frame id: everything before the last slash.

    The same rule as `latplan.util.adjacency.parse_frame_id`, repeated here so
    that this module stays free of the `latplan` package import.
    """
    return str(frame_id).rpartition("/")[0]


def assign_clips(frame_ids, fractions=(0.90, 0.05, 0.05)):
    """`{clip: "train" | "val" | "test"}`, clips in first-appearance order."""
    clips = list(collections.OrderedDict.fromkeys(_video_of(f)
                                                  for f in frame_ids))
    if len(clips) < 3:
        raise ValueError("SPLIT=clip needs at least 3 clips, one for each of "
                         "train, val and test; this load has %d" % len(clips))
    n_val = max(1, int(round(fractions[1] * len(clips))))
    n_test = max(1, int(round(fractions[2] * len(clips))))
    n_train = len(clips) - n_val - n_test
    out = collections.OrderedDict()
    for k, clip in enumerate(clips):
        out[clip] = ("train" if k < n_train
                     else "val" if k < n_train + n_val else "test")
    return out


def split_states(states, frame_ids, pairs, fractions=(0.90, 0.05, 0.05)):
    """`(train, val, test, info)`, each split the states of its own clips' pairs.

    `states` is indexed like `frame_ids`. `info` holds the clip and pair counts
    of each split, for the log and the run manifest.
    """
    assignment = assign_clips(frame_ids, fractions)
    by_split = dict((s, []) for s in SPLITS)
    for i, j in pairs:
        by_split[assignment[_video_of(frame_ids[i])]].append((i, j))
    for name in SPLITS:
        if not by_split[name]:
            raise ValueError("SPLIT=clip left the %s split with no pair of "
                             "consecutive frames" % name)
    out = []
    for name in SPLITS:
        before = [i for i, _ in by_split[name]]
        after = [j for _, j in by_split[name]]
        out.append(np.concatenate([states[before], states[after]]))
    info = {
        "clips": dict((s, sum(1 for v in assignment.values() if v == s))
                      for s in SPLITS),
        "pairs": dict((s, len(by_split[s])) for s in SPLITS),
        "val_clips": [c for c, s in assignment.items() if s == "val"],
        "test_clips": [c for c, s in assignment.items() if s == "test"],
    }
    return out[0], out[1], out[2], info


def frames_of(frame_ids, clips):
    """Indices of the frames whose clip is in `clips`, in load order.

    `tools/replot.py` uses it to draw its test pictures from the test clips a
    run's manifest names, rather than from any state of the npz.
    """
    wanted = set(clips)
    return [k for k, f in enumerate(frame_ids) if _video_of(f) in wanted]
