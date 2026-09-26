#!/usr/bin/env bash
#
# moma_change_rate.sbatch — MOMA-LRG's change rate, which it never had.
#
# MOMA-LRG was closed on 2026-09-22 for holding 2 object-to-object relations
# of 985,243. The author's ruling of 2026-09-23 voided that column, because an
# object is what object detection calls an object and `person` is a class. So
# all 985,243 count and the row reopened with NO RATE AT ALL.
#
# ITS UNIT IS NOT A TRACK AND THAT MATTERS. A relationship sits inside a
# `higher_order_interaction` which carries a single `time`, and the same edge
# repeats at time 6, 7 and 8. So the pair series is built over the interaction
# times of one sub-activity, which is the closest thing MOMA has to a
# temporal sequence.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/moma-rate-%j.out
#SBATCH --error=logs/moma-rate-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
python3 - <<'PY'
import json, collections

d = json.load(open('data/video/moma_lrg/raw/anns/anns.json'))
tax = json.load(open('data/video/moma_lrg/raw/anns/taxonomy/relationship.json'))
group = {}
for g, rows in tax.items():
    for r in rows:
        group[r[0]] = g

# An `attentional` predicate names what an actor does. A `spatial` one names
# where two entities are. The author's ruling says what an OBJECT is and says
# nothing about what a PREDICATE means, so the split this project uses
# elsewhere applies here too.
def is_action(p):
    return group.get(p) == 'attentional'

pairs = collections.defaultdict(list)
rows = 0
snapshots = 0
preds = collections.Counter()
for video in d:
    act = video.get('activity') or {}
    for sact in act.get('sub_activities', []):
        key0 = (video.get('file_name'), sact.get('id'))
        for hoi in sact.get('higher_order_interactions', []):
            snapshots += 1
            t = hoi.get('time')
            per = collections.defaultdict(set)
            for rel in hoi.get('relationships', []):
                rows += 1
                preds[rel['class_name']] += 1
                per[(key0, rel['source_id'], rel['target_id'])].add(
                    rel['class_name'])
            for k, v in per.items():
                pairs[k].append((t, frozenset(v)))

single = same = change = act_n = 0
for v in pairs.values():
    v.sort()
    sets = [s for _, s in v]
    if len(sets) == 1:
        single += 1
    elif len(set(sets)) == 1:
        same += 1
    else:
        change += 1
    if any(is_action(p) for s in sets for p in s):
        act_n += 1
n = len(pairs) or 1
print('MOMA-LRG, change rate, ALL PAIRS')
print('  videos                      %d' % len(d))
print('  interaction snapshots       %d' % snapshots)
print('  relationship rows           %d' % rows)
print('  PAIRS, one (video, sub-activity, src, tgt) each  %d' % len(pairs))
print('   seen at one time only      %7d  (%.1f%%)' % (single, 100.0*single/n))
print('   many times, SAME predicate %7d  (%.1f%%)' % (same, 100.0*same/n))
print('   predicate CHANGES          %7d  (%.1f%%)' % (change, 100.0*change/n))
print('   an ATTENTIONAL predicate   %7d  (%.1f%%)' % (act_n, 100.0*act_n/n))
print()
print('  predicates:')
for k, v in preds.most_common():
    print('    %-34s %8d   (%s)' % (k, v, group.get(k, '?')))
PY
