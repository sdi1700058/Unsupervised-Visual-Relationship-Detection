#!/usr/bin/env bash
#
# change_is_succession.sbatch — recompute CHANGE as succession, not co-occurrence.
#
#   sbatch --job-name=succession sh/change_is_succession.sbatch
#
# WHY. An adversarial audit on 2026-09-26 found that this project's "change"
# column asks the wrong question of VidVRD, VidOR and PVSG. Those three ship
# a relation as an INTERVAL, `predicate` with `begin_fid` and `end_fid`. The
# measurement asked whether a pair ever carries two distinct predicates over
# its life. It never asked whether the two intervals OVERLAP.
#
# A dog that is `sit_left` of a sofa AND `watch` ing it at the same instant
# has one state, not two. Verified on the login node before this job was
# written: of VidVRD's 5,354 "changing" directed pairs, 4,164 (77.8%) have
# every cross-predicate interval overlapping, and only 118 (2.2%) are pure
# succession. The corrected VidVRD rate is 25.9 per cent directed, against
# the 70.1 that put it forward as the easy rung.
#
# KIT, MOMA-LRG, Endoscapes and EASG are NOT affected, and the adversary
# checked all four: each aggregates to a per-timestep snapshot first and then
# compares consecutive snapshots, which is the correct shape. This job gives
# the other three the same treatment.
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
import json, glob, collections

def snapshot_change(spans):
    """True when the predicate SET actually moves between two instants.

    The frame boundaries of every interval are the only instants where the
    set can change, so they are the only ones tested. Two predicates true at
    one instant are ONE state.
    """
    edges = sorted({b for _, b, _ in spans} | {e for _, _, e in spans})
    prev = None
    for t in edges:
        state = frozenset(p for p, b, e in spans if b <= t < e)
        if not state:
            continue
        if prev is not None and state != prev:
            return True
        prev = state
    return False


def report(name, pairs):
    n = len(pairs) or 1
    old = sum(1 for v in pairs.values() if len({p for p, _, _ in v}) > 1)
    new = sum(1 for v in pairs.values() if snapshot_change(v))
    print("%-22s pairs %7d   OLD change %7d (%.1f%%)   SUCCESSION %7d (%.1f%%)"
          % (name, len(pairs), old, 100.0 * old / n, new, 100.0 * new / n),
          flush=True)


# ---- VidVRD, as a control: the login node already answered 25.9 and 29.7 ----
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
    report(tag + " undirected", undirected)

# ---- PVSG: relations are segments of [subject, object, predicate, [ranges]] ----
try:
    pv = json.load(open("data/video/pvsg/raw/pvsg.json"))
except Exception as exc:
    print("pvsg: %s" % exc)
else:
    pairs = collections.defaultdict(list)
    for vid in pv.get("data", []):
        vid_id = vid.get("video_id")
        for rel in vid.get("relations", []) or []:
            subj, obj, pred, spans = rel[0], rel[1], rel[2], rel[3]
            for span in spans:
                pairs[(vid_id, subj, obj)].append((pred, span[0], span[1]))
    report("pvsg directed", pairs)
PY
