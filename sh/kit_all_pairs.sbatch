#!/usr/bin/env bash
#
# kit_all_pairs.sbatch — KIT's change rate with the HANDS included.
#
# The author ruled on 2026-09-23 that an object is what object detection calls
# an object. KIT's own object table names `LeftHand` and `RightHand` as
# classes 14 and 15, so they are objects and the earlier measurement, which
# dropped them by name, described the wrong population exactly as every other
# rate did.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/kitall-%j.out
#SBATCH --error=logs/kitall-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
python3 - <<'PY'
import json, os, glob, collections
ROOT = 'data/video/kit_bimanual/ext/bimacs_derived_data'
rel_files = glob.glob(os.path.join(ROOT, '*', '*', '*', 'spatial_relations', '*.json'))
print('relation frame files: %d' % len(rel_files))

def take_of(p):
    return os.sep.join(p.split(os.sep)[:-2])

# A DYNAMIC predicate by the distributor's own marking, minus `stable`, which
# the taxonomy marks dynamic and which means the distance did NOT change.
MOVING = {'moving together', 'halting together', 'fixed moving together',
          'getting close', 'moving apart'}

pairseq = collections.defaultdict(list)
rel = collections.Counter()
rows = 0
for p in sorted(rel_files):
    t = take_of(p)
    try:
        d = json.load(open(p))
    except Exception:
        continue
    per = collections.defaultdict(set)
    for r in d:
        rows += 1
        rel[r.get('relation_name')] += 1
        per[(t, r.get('subject_index'), r.get('object_index'))].add(
            r.get('relation_name'))
    for k, v in per.items():
        pairseq[k].append(frozenset(v))

single = same = change = moving = 0
for seq in pairseq.values():
    if len(seq) == 1:
        single += 1
    elif len(set(seq)) == 1:
        same += 1
    else:
        change += 1
    if any(MOVING & s for s in seq):
        moving += 1
n = len(pairseq) or 1
print()
print('KIT Bimanual -- ALL PAIRS, hands INCLUDED')
print('  relation rows                 %d' % rows)
print('  PAIRS                         %d' % len(pairseq))
print('   one frame only               %8d  (%.1f%%)' % (single, 100.0*single/n))
print('   many frames, SAME relations  %8d  (%.1f%%)' % (same, 100.0*same/n))
print('   relations CHANGE             %8d  (%.1f%%)' % (change, 100.0*change/n))
print('   a MOVING predicate appears   %8d  (%.1f%%)' % (moving, 100.0*moving/n))
PY
