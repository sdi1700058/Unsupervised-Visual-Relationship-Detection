"""Which frame pairs are one step apart, and how many are not.

**The defect this exists to close.** `build_transitions` paired frame `i` with
frame `i+1` **in the loaded array** and compared only the video id, never the
frame number. Both video loaders skip frames in silence -- a missing jpg, an
empty annotation -- so every skip became an "adjacent" pair.

Measured on the production configuration, 30fps, mo3, p8, 60 videos: **45 of
4,335 emitted transitions span 16 to 271 real frames**, up to nine seconds, as
one action step. At 3fps every transition spans ten frames, because only one
annotated frame in ten has a matching file. FOSAE's action model assumes a
transition is one step, and the assumption was being broken without a word.

**What was decided, 2026-09-06.** Count always; refuse only under
`STRICT_ADJACENCY=1`. The default therefore reproduces every existing run
exactly, so measurements stay comparable, while the loss stops being invisible.
A clean run is available by setting the variable, and the counts go into the
export so nobody has to guess which was which.

**What is deliberately not refused.** A pair whose frame numbers cannot be
parsed is kept even under `STRICT_ADJACENCY=1`, and counted as `unknown_gap`.
Refusing on ignorance would silently discard whole datasets whose ids are
formatted differently, which is the same disease with the opposite sign.

This module imports nothing from `latplan`, so it can be read without the
training stack. Standard library only, Python 3.6 clean.
"""

import os


def parse_frame_id(frame_id):
    """`(video, frame_number)` from a frame id, number `None` if unreadable.

    VidVRD writes `<video>/<NNNNNN>` and Action Genome writes
    `<vid>.mp4/<NNNNNN>.png`, so the split is on the **last** slash and the
    number is whatever leading digits follow it.
    """
    text = str(frame_id)
    if "/" not in text:
        return text, None
    video, _, tail = text.rpartition("/")
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return video, int(digits) if digits else None


def strict_from_env(env=None):
    """Whether `STRICT_ADJACENCY=1` is set. Off unless asked."""
    return (env or os.environ).get("STRICT_ADJACENCY", "0") == "1"


def sequential_pairs(frame_ids, strict=False):
    """`([(i, j)], stats)` for consecutive frames of the same video.

    `stats` carries `pairs`, `boundaries`, `non_adjacent`, `unknown_gap`,
    `max_gap`, `refused` and `strict`, so a caller can record what the load
    actually contained rather than only how many rows came out.
    """
    stats = {"pairs": 0, "boundaries": 0, "non_adjacent": 0,
             "unknown_gap": 0, "max_gap": 0, "refused": 0,
             "strict": bool(strict)}
    pairs = []
    for i in range(len(frame_ids) - 1):
        video_a, frame_a = parse_frame_id(frame_ids[i])
        video_b, frame_b = parse_frame_id(frame_ids[i + 1])
        if video_a != video_b:
            stats["boundaries"] += 1
            continue

        if frame_a is None or frame_b is None:
            stats["unknown_gap"] += 1
            pairs.append((i, i + 1))
            continue

        gap = frame_b - frame_a
        if gap != 1:
            stats["non_adjacent"] += 1
            stats["max_gap"] = max(stats["max_gap"], gap)
            if strict:
                stats["refused"] += 1
                continue
        pairs.append((i, i + 1))

    stats["pairs"] = len(pairs)
    return pairs, stats


def describe(stats):
    """One line for a log, saying what the load lost."""
    if not stats.get("non_adjacent") and not stats.get("unknown_gap"):
        return ("%d transitions, all one frame apart"
                % stats.get("pairs", 0))

    parts = ["%d transitions" % stats.get("pairs", 0)]
    if stats.get("non_adjacent"):
        parts.append("%d span more than one frame, worst %d"
                     % (stats["non_adjacent"], stats.get("max_gap", 0)))
    if stats.get("refused"):
        parts.append("%d refused by STRICT_ADJACENCY" % stats["refused"])
    elif stats.get("non_adjacent"):
        parts.append("kept, because a transition that is not one step still "
                     "trains as though it were; set STRICT_ADJACENCY=1 to "
                     "drop them")
    if stats.get("unknown_gap"):
        parts.append("%d with unreadable frame numbers, kept"
                     % stats["unknown_gap"])
    return "; ".join(parts)
