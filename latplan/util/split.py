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
import re

import numpy as np

SPLITS = ("train", "val", "test")


def _video_of(frame_id):
    """The clip of a frame id: everything before the last slash.

    The same rule as `latplan.util.adjacency.parse_frame_id`, repeated here so
    that this module stays free of the `latplan` package import.
    """
    return str(frame_id).rpartition("/")[0]


def _group_of(clip, group):
    """The group of a clip: what `group`, a regex, matches at its start.

    A clip the regex does not match is its own group, so a pattern written
    for one dataset leaves the clips of another as they were.
    """
    found = re.match(group, clip) if group else None
    return found.group(0) if found else clip


def _assign_groups(clips, group, n_val, n_test, val_groups=None,
                   test_groups=None):
    """Whole groups: the named ones, or else from the end, test and then
    val, each until it holds its clip count. At least one group stays for
    each split."""
    groups = collections.OrderedDict()
    for clip in clips:
        groups.setdefault(_group_of(clip, group), []).append(clip)
    if len(groups) < 3:
        raise ValueError("SPLIT_GROUP=%r leaves %d groups; train, val and "
                         "test need one each" % (group, len(groups)))
    order = list(groups)
    split_of = dict((g, "train") for g in order)
    if val_groups or test_groups:
        for name, named in (("val", val_groups), ("test", test_groups)):
            missing = [g for g in (named or []) if g not in groups]
            if not named or missing:
                raise ValueError("SPLIT_%s names %r, and the load holds "
                                 "the groups %r" % (name.upper(), named,
                                                    order))
            for g in named:
                split_of[g] = name
        out = collections.OrderedDict()
        for clip in clips:
            out[clip] = split_of[_group_of(clip, group)]
        return out
    k = len(order) - 1
    for name, need, keep in (("test", n_test, 2), ("val", n_val, 1)):
        got = 0
        while k >= keep and got < need:
            split_of[order[k]] = name
            got += len(groups[order[k]])
            k -= 1
    out = collections.OrderedDict()
    for clip in clips:
        out[clip] = split_of[_group_of(clip, group)]
    return out


def assign_clips(frame_ids, fractions=(0.90, 0.05, 0.05), group=None,
                 val_groups=None, test_groups=None):
    """`{clip: "train" | "val" | "test"}`, clips in first-appearance order.

    `group`, a regex such as `S[0-9]+`, keeps the clips whose names start
    with one match in one split, so that val and test show sessions that
    training never saw. Added 2026-10-05: HA-ViD windows load in sorted name
    order, and the val windows of the 2026-10-03 runs came from session S25,
    which training also held. `val_groups` and `test_groups` name the
    groups of val and test, because the largest HA-ViD session, S25, sorts
    second to last.
    """
    clips = list(collections.OrderedDict.fromkeys(_video_of(f)
                                                  for f in frame_ids))
    if len(clips) < 3:
        raise ValueError("SPLIT=clip needs at least 3 clips, one for each of "
                         "train, val and test; this load has %d" % len(clips))
    n_val = max(1, int(round(fractions[1] * len(clips))))
    n_test = max(1, int(round(fractions[2] * len(clips))))
    if group:
        return _assign_groups(clips, group, n_val, n_test, val_groups,
                              test_groups)
    n_train = len(clips) - n_val - n_test
    out = collections.OrderedDict()
    for k, clip in enumerate(clips):
        out[clip] = ("train" if k < n_train
                     else "val" if k < n_train + n_val else "test")
    return out


def split_states(states, frame_ids, pairs, fractions=(0.90, 0.05, 0.05),
                 group=None, val_groups=None, test_groups=None):
    """`(train, val, test, info)`, each split the states of its own clips' pairs.

    `states` is indexed like `frame_ids`. `info` holds the clip and pair counts
    of each split, for the log and the run manifest.
    """
    assignment = assign_clips(frame_ids, fractions, group=group,
                              val_groups=val_groups, test_groups=test_groups)
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
        "group": group,
    }
    return out[0], out[1], out[2], info


def frames_of(frame_ids, clips):
    """Indices of the frames whose clip is in `clips`, in load order.

    `tools/replot.py` uses it to draw its test pictures from the test clips a
    run's manifest names, rather than from any state of the npz.
    """
    wanted = set(clips)
    return [k for k, f in enumerate(frame_ids) if _video_of(f) in wanted]
