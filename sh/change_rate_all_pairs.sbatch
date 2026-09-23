#!/usr/bin/env bash
#
# change_rate_all_pairs.sbatch — the change rate with NO person exclusion.
#
# THE AUTHOR RULED ON 2026-09-23, in their own words: "when we say object we
# are referring to the term used in computer vision and specifically object
# detection fields". In object detection `person` is a CLASS, exactly as
# `bicycle` is. So the "object-to-object" column this project sorted on, which
# excluded person and animal subjects, was a distinction the field does not
# make and this session invented.
#
# EVERY CHANGE RATE ON RECORD WAS COMPUTED OVER THE PERSON-EXCLUDED SUBSET, so
# every one of them is a statistic about the wrong population. This job
# recomputes them over ALL pairs.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/allpairs-%j.out
#SBATCH --error=logs/allpairs-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
python3 - <<'PY'
import json, glob, collections

# Which predicates name an ACTION rather than a viewpoint. Unchanged from the
# 2026-09-22 split, because the author's ruling is about what an OBJECT is and
# says nothing about what a predicate means.
VIEW = set(('next_to', 'in_front_of', 'behind', 'beneath', 'above', 'inside',
            'away', 'towards', 'larger', 'taller', 'faster', 'front', 'left',
            'right', 'close', 'beside', 'over', 'near', 'past'))
STATIC_ROOT = set(('stand', 'sit', 'lie', 'stop', 'past'))


def is_action(p):
    if p in VIEW:
        return False
    base = p.split('_')[0]
    if base in STATIC_ROOT:
        return False
    if base in ('walk', 'run', 'fly', 'jump', 'swim', 'creep', 'move'):
        return True
    return True


def survey(name, files):
    pairs = collections.defaultdict(set)
    total = 0
    preds = collections.Counter()
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d.get('relation_instances', []):
            total += 1
            preds[r['predicate']] += 1
            pairs[(f, r['subject_tid'], r['object_tid'])].add(r['predicate'])
    single = same = change = act = 0
    for v in pairs.values():
        if len(v) == 1:
            single += 1
        else:
            change += 1
        if any(is_action(p) for p in v):
            act += 1
    n = len(pairs) or 1
    print('%s  -- ALL PAIRS, no person exclusion' % name)
    print('  relation instances          %8d' % total)
    print('  distinct predicates         %8d' % len(preds))
    print('  PAIRS                       %8d' % len(pairs))
    print('   one predicate only         %8d  (%.1f%%)' % (single, 100.0*single/n))
    print('   predicate CHANGES          %8d  (%.1f%%)' % (change, 100.0*change/n))
    print('   an ACTION predicate present%8d  (%.1f%%)' % (act, 100.0*act/n))
    print()


survey('VidVRD', sorted(glob.glob('data/video/vidvrd/annotations/*/*.json')))
survey('VidOR', sorted(glob.glob('data/video/vidor/annotations/**/*.json',
                                 recursive=True)))

# PVSG uses a different schema.
d = json.load(open('data/video/pvsg/raw/pvsg.json'))
pairs = collections.defaultdict(set)
segs = 0
for rec in d['data']:
    for r in rec['relations']:
        segs += len(r[3])
        pairs[(rec['video_id'], r[0], r[1])].add(r[2])
chg = sum(1 for v in pairs.values() if len(v) > 1)
n = len(pairs) or 1
print('PVSG  -- ALL PAIRS, no person exclusion')
print('  segments                    %8d' % segs)
print('  PAIRS                       %8d' % len(pairs))
print('   predicate CHANGES          %8d  (%.1f%%)' % (chg, 100.0*chg/n))
PY
