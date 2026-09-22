#!/usr/bin/env bash
#
# count_vidor_relations.sbatch — the object-to-object split for VidOR.
#
# WHY THIS EXISTS. MOMA-LRG showed on 2026-09-22 that a headline relation
# count says nothing about how many relations join two OBJECTS: 985,243 total
# and 2 object-to-object. A 150-file sample of VidOR answered 14.0 per cent
# object-to-object against VidVRD's measured 75.6, so the full number decides
# which of this project's own locked datasets carries the relations it needs.
#
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/vidor-rel-%j.out
#SBATCH --error=logs/vidor-rel-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
python3 - <<'PY'
import json, glob, collections
files = sorted(glob.glob('data/video/vidor/annotations/**/*.json', recursive=True))
PERSON = {'adult', 'child', 'baby'}
tot = oo = po = 0
pred = collections.Counter(); oopred = collections.Counter()
cats = collections.Counter(); oocats = collections.Counter()
splits = collections.Counter()
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    splits['training' if '/training' in f else 'validation'] += 1
    tid2cat = {t['tid']: t['category'] for t in d.get('subject/objects', [])}
    for v in tid2cat.values():
        cats[v] += 1
    for r in d.get('relation_instances', []):
        tot += 1
        s = tid2cat.get(r['subject_tid'], '?')
        o = tid2cat.get(r['object_tid'], '?')
        p = r['predicate']; pred[p] += 1
        if s.lower() in PERSON or o.lower() in PERSON:
            po += 1
        else:
            oo += 1; oopred[p] += 1
            oocats[s] += 1; oocats[o] += 1
print('VidOR, FULL count, measured on this disk')
print('  annotation files     %d   %s' % (len(files), dict(splits)))
print('  relation instances   %d' % tot)
print('  OBJECT to OBJECT     %d  (%.1f%%)' % (oo, 100.0*oo/tot if tot else 0))
print('  involves a person    %d  (%.1f%%)' % (po, 100.0*po/tot if tot else 0))
print('  distinct predicates  %d' % len(pred))
print('  distinct categories  %d' % len(cats))
print()
print('OBJECT-TO-OBJECT PREDICATES')
for k, v in oopred.most_common(): print('  %-18s %8d' % (k, v))
print()
print('CATEGORIES APPEARING IN AN OBJECT-TO-OBJECT EDGE')
for k, v in oocats.most_common(25): print('  %-18s %8d' % (k, v))
PY
