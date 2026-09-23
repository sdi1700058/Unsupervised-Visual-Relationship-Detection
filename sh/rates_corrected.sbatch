#!/usr/bin/env bash
#
# rates_corrected.sbatch — two adversary defects fixed in one pass.
#
# DEFECT A, and it is a bug in this project's own classifier. is_action put
# `towards` and `away` in the VIEW set. VidOR's ONLY vocabulary for approach
# and departure is those two bare words, 31,525 instances or 10.6 per cent of
# everything it annotates, while VidVRD says `walk_toward` and `run_away` and
# scores as action through its verb base. So the classifier penalised VidOR
# for a vocabulary SHAPE and the 20.9 against 58.4 comparison was never apples
# to apples.
#
# The principle, stated so the next reader can argue with it: a predicate is
# an ACTION when it asserts a CHANGE in the spatial configuration. `towards`,
# `away` and `past` do. `next_to`, `above` and `inside` do not. `larger`,
# `taller` and `faster` compare a property rather than a configuration.
#
# DEFECT B. 99.6 per cent of VidVRD's pairs have a same-file reverse-direction
# counterpart, so one physical dyad is counted as two. Both conventions are
# reported rather than one chosen silently.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/ratescorr-%j.out
#SBATCH --error=logs/ratescorr-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
python3 - <<'PY'
import json, glob, collections

# A CONFIGURATION, not a change.
STATIC = set(('next_to', 'in_front_of', 'behind', 'beneath', 'above', 'below',
              'inside', 'front', 'left', 'right', 'close', 'beside', 'over',
              'near', 'surround', 'outside'))
# A PROPERTY comparison, not a configuration.
COMPARE = set(('larger', 'taller', 'faster', 'smaller', 'shorter', 'slower'))
# Roots that describe a POSE rather than a motion.
POSE = set(('stand', 'sit', 'lie', 'stop'))
# Bare words that DO assert a change. This is the correction.
MOTION = set(('towards', 'toward', 'away', 'past'))


def is_action(p):
    if p in MOTION:
        return True
    if p in STATIC or p in COMPARE:
        return False
    base = p.split('_')[0]
    if base in POSE:
        return False
    if base in STATIC or base in COMPARE:
        return False
    return True


def survey(name, files):
    directed = collections.defaultdict(set)
    undirected = collections.defaultdict(set)
    total = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d.get('relation_instances', []):
            total += 1
            a, b = r['subject_tid'], r['object_tid']
            directed[(f, a, b)].add(r['predicate'])
            undirected[(f, min(a, b), max(a, b))].add(r['predicate'])
    for label, pairs in (('DIRECTED', directed), ('UNDIRECTED dyads', undirected)):
        chg = sum(1 for v in pairs.values() if len(v) > 1)
        act = sum(1 for v in pairs.values() if any(is_action(p) for p in v))
        n = len(pairs) or 1
        print('%-8s %-18s pairs %7d   change %7d (%.1f%%)   action %7d (%.1f%%)'
              % (name, label, len(pairs), chg, 100.0*chg/n, act, 100.0*act/n))
    print('%-8s relation instances %d' % (name, total))
    print()


survey('VidVRD', sorted(glob.glob('data/video/vidvrd/annotations/*/*.json')))
survey('VidOR', sorted(glob.glob('data/video/vidor/annotations/**/*.json',
                                 recursive=True)))
PY
