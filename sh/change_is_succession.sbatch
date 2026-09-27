#!/usr/bin/env bash
#
# change_is_succession.sbatch — CHANGE as succession, under THREE definitions.
#
#   sbatch --job-name=succession sh/change_is_succession.sbatch
#
# WHY. An adversarial audit on 2026-09-26 found that this project's "change"
# column asks the wrong question of VidVRD, VidOR and PVSG. Those three ship
# a relation as an INTERVAL, `predicate` with `begin_fid` and `end_fid`. The
# original measurement asked whether a pair ever carries two distinct
# predicates over its life. It never asked whether the two intervals OVERLAP.
#
# A dog `sit_left` of a sofa AND `watch`ing it at one instant has one state,
# not two. VidVRD's rate fell from 70.1 per cent to 25.9 when that was fixed.
#
# AND THEN A SECOND ADVERSARY FOUND THE FIX'S OWN HOLE, which is why this
# script now reports THREE numbers rather than one. The snapshot walk skipped
# an empty state, so a relation that DISAPPEARS and later returns in the same
# state read as unchanged. Verified over all 1,000 VidVRD files: 2,175 pairs
# hold a true inner gap and 923 of them, 12.1 per cent of all pairs, were
# judged unchanged by that skip. Counting a disappearance as a change takes
# VidVRD from 25.9 to 38.0 per cent.
#
# NEITHER NUMBER IS THE TRUE ONE, AND THAT IS THE POINT. Whether a relation
# ceasing to hold is a transition is a MODELLING choice this project has
# never written down. `build_transitions` pairs frame t with t+1, and a pair
# with no relation at t+1 is a real state of the world. So the column reports
# both bounds and names the choice rather than hiding it inside a loop.
#
# WHAT ELSE MOVED HERE. Two adversaries pointed out that the mechanism
# numbers, the interval lengths and the all-identical share, also lived in a
# throwaway heredoc. They are computed here now. And PVSG's spans are CLOSED,
# which an adversary measured: `num_frames` is 180 at 5 fps over 36.0 seconds
# and the largest span value is 179. This script used a half-open test on it,
# which was a real defect worth exactly four pairs.
#
# NOT AFFECTED, and an adversary read all four sources rather than trusting
# the claim: `sh/kit_all_pairs.sbatch`, `sh/moma_change_rate.sbatch`,
# `workbench/tools/endoscapes_relations.py` and
# `workbench/tools/easg_relations.py` each aggregate every predicate true at
# one instant into ONE set before comparing, so the overlap defect cannot
# arise in them. Whether they share the GAP blind spot is still open.
#
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}" || exit 2
module load python/3.12.1

python3 - <<'PY'
import json, glob, collections, statistics


def states(spans, closed=False):
    """The predicate SET at every instant where it can change.

    An interval's endpoints are the only instants where the set moves, so
    they are the only ones tested. Two predicates true at one instant are
    ONE state, which is the whole point.
    """
    edges = sorted({b for _, b, _ in spans} | {e for _, _, e in spans})
    if closed:
        return [frozenset(p for p, b, e in spans if b <= t <= e) for t in edges]
    return [frozenset(p for p, b, e in spans if b <= t < e) for t in edges]


def verdicts(spans, closed=False):
    """(moved_ignoring_gaps, has_inner_gap).

    A leading or trailing empty state is an artefact of walking endpoints,
    not a gap. Only an empty state BETWEEN two non-empty ones is one.
    """
    seq = states(spans, closed)
    filled = [i for i, s in enumerate(seq) if s]
    if not filled:
        return False, False
    inner = seq[filled[0]:filled[-1] + 1]
    nonempty = [s for s in inner if s]
    moved = any(a != b for a, b in zip(nonempty, nonempty[1:]))
    return moved, any(not s for s in inner)


def report(name, pairs, closed=False):
    n = len(pairs) or 1
    old = sum(1 for v in pairs.values() if len({p for p, _, _ in v}) > 1)
    moved = gapped = hidden = 0
    for v in pairs.values():
        m, g = verdicts(v, closed)
        if m:
            moved += 1
        if g:
            gapped += 1
            if not m:
                hidden += 1
    print("%-20s pairs %7d" % (name, len(pairs)))
    print("    OLD, any two distinct predicates   %7d (%.1f%%)"
          % (old, 100.0 * old / n))
    print("    SUCCESSION, gaps ignored           %7d (%.1f%%)   <- lower bound"
          % (moved, 100.0 * moved / n))
    print("    ... plus a disappearance as change %7d (%.1f%%)   <- upper bound"
          % (moved + hidden, 100.0 * (moved + hidden) / n))
    print("    pairs holding a true inner gap     %7d (%.1f%%), of which %d were hidden"
          % (gapped, 100.0 * gapped / n, hidden), flush=True)


def mechanism(name, pairs):
    """Why VidVRD collapses and VidOR does not: the annotation GRAIN."""
    multi = [v for v in pairs.values() if len({p for p, _, _ in v}) > 1]
    identical = sum(1 for v in multi if len({(b, e) for _, b, e in v}) == 1)
    lengths = sorted(e - b for v in pairs.values() for _, b, e in v)
    if not lengths or not multi:
        return
    print("    interval length frames: p50 %d  p90 %d  max %d"
          % (statistics.median(lengths), lengths[int(len(lengths) * 0.9)],
             lengths[-1]))
    print("    multi-predicate pairs whose intervals are ALL IDENTICAL "
          "%d of %d (%.1f%%)"
          % (identical, len(multi), 100.0 * identical / len(multi)), flush=True)


for tag, root in (("vidvrd", "data/video/vidvrd/annotations/*/*.json"),
                  ("vidor", "data/video/vidor/annotations/*/*/*.json")):
    directed = collections.defaultdict(list)
    for fp in sorted(glob.glob(root)):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        for r in d.get("relation_instances", []):
            directed[(fp, r["subject_tid"], r["object_tid"])].append(
                (r["predicate"], r["begin_fid"], r["end_fid"]))
    undirected = collections.defaultdict(list)
    for (fp, s, o), v in directed.items():
        undirected[(fp, min(s, o), max(s, o))] += v
    report(tag + " directed", directed)
    mechanism(tag, directed)
    report(tag + " undirected", undirected)

# PVSG spans are CLOSED: num_frames 180 at 5 fps over 36.0 s, max span 179.
try:
    pv = json.load(open("data/video/pvsg/raw/pvsg.json"))
except Exception as exc:
    print("pvsg: %s" % exc)
else:
    pairs = collections.defaultdict(list)
    for vid in pv.get("data", []):
        for rel in vid.get("relations", []) or []:
            for span in rel[3]:
                pairs[(vid.get("video_id"), rel[0], rel[1])].append(
                    (rel[2], span[0], span[1]))
    report("pvsg directed", pairs, closed=True)
PY
