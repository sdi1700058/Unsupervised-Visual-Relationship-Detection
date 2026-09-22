#!/usr/bin/env bash
#
# measure_kit.sbatch — the three numbers a dataset row needs, for KIT Bimanual.
#
# WHY A JOB. The relations are ONE JSON PER FRAME, so 540 recordings come to
# roughly 220,000 small files. Walking that on the login node is exactly the
# metadata hammering the cluster policy forbids.
#
# THE THREE NUMBERS, set on 2026-09-22 after an adversary showed the first two
# are not enough: how many relations, how many join two non-persons, and HOW
# MANY OF THOSE EVER CHANGE. KIT lists OpenPose hands as objects, so the hand
# indices are this dataset's persons and the split turns on finding them.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/kit-rel-%j.out
#SBATCH --error=logs/kit-rel-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
python3 - <<'PY'
import json, os, glob, collections
ROOT = 'data/video/kit_bimanual/ext/bimacs_derived_data'
rel_files = glob.glob(os.path.join(ROOT, '*', '*', '*', 'spatial_relations', '*.json'))
obj_files = glob.glob(os.path.join(ROOT, '*', '*', '*', '3d_objects', '*.json'))
print('spatial_relations frame files: %d' % len(rel_files))
print('3d_objects frame files       : %d' % len(obj_files))

# index -> class name, per take, read from the object file of the same frame
def take_of(p):
    parts = p.split(os.sep)
    return os.sep.join(parts[:-2])

names_by_take = {}
for p in obj_files:
    t = take_of(p)
    if t in names_by_take:
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if isinstance(d, list):
        names_by_take[t] = {i: (o.get('class_name') or o.get('name') or '?')
                            for i, o in enumerate(d)}
print('takes with an object name table: %d' % len(names_by_take))
sample = list(names_by_take.items())[:2]
for t, n in sample:
    print('  %s -> %s' % (os.sep.join(t.split(os.sep)[-3:]), n))

HANDISH = ('hand', 'left hand', 'right hand', 'lefthand', 'righthand')
rel = collections.Counter()
classes = collections.Counter()
tot = oo = hand = 0
# pair -> ordered list of relation SETS, one per frame, to see change
pairseq = collections.defaultdict(list)
frames_per_take = collections.Counter()

for p in sorted(rel_files):
    t = take_of(p)
    names = names_by_take.get(t, {})
    try:
        d = json.load(open(p))
    except Exception:
        continue
    frames_per_take[t] += 1
    per_pair = collections.defaultdict(set)
    for r in d:
        s = r.get('subject_index'); o = r.get('object_index')
        n = r.get('relation_name')
        tot += 1; rel[n] += 1
        sn = str(names.get(s, s)).lower(); on = str(names.get(o, o)).lower()
        classes[sn] += 1
        if any(h in sn for h in HANDISH) or any(h in on for h in HANDISH):
            hand += 1
        else:
            oo += 1
            per_pair[(t, s, o)].add(n)
    for k, v in per_pair.items():
        pairseq[k].append(frozenset(v))

single = samep = change = 0
for k, seq in pairseq.items():
    uniq = set(seq)
    if len(seq) == 1:
        single += 1
    elif len(uniq) == 1:
        samep += 1
    else:
        change += 1
n = len(pairseq) or 1
print()
print('KIT Bimanual, measured')
print('  relation rows (per frame)     %d' % tot)
print('  involves a HAND               %d  (%.1f%%)' % (hand, 100.0*hand/tot if tot else 0))
print('  OBJECT to OBJECT              %d  (%.1f%%)' % (oo, 100.0*oo/tot if tot else 0))
print('  distinct relation names       %d' % len(rel))
print('  takes                         %d' % len(frames_per_take))
print()
print('  object-object PAIRS           %d' % len(pairseq))
print('   one frame only               %6d  (%.1f%%)' % (single, 100.0*single/n))
print('   many frames, SAME relations  %6d  (%.1f%%)' % (samep, 100.0*samep/n))
print('   relations CHANGE             %6d  (%.1f%%)' % (change, 100.0*change/n))
print()
print('  EVERY RELATION NAME')
for k, v in rel.most_common():
    print('    %-24s %10d' % (k, v))
print()
print('  object classes seen (top 20)')
for k, v in classes.most_common(20):
    print('    %-24s %10d' % (k, v))
PY
