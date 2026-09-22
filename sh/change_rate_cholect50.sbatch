#!/usr/bin/env bash
#
# change_rate_cholect50.sbatch — the third number for CholecT50.
#
# A row needs three numbers since 2026-09-22: relations, object-to-object, and
# HOW MANY OF THOSE EVER CHANGE. VidVRD answers 69.9 per cent, VidOR 22.1 and
# PVSG 4.7, and the third number reversed the reading of the first two.
#
# CholecT50 HAS NO INSTANCE IDS, so a pair here is (video, instrument, target)
# at CATEGORY level. That is a weaker pair than VidVRD's (video, tid, tid) and
# the number is not directly comparable. It is computed anyway, because a
# category-level pair whose VERB never changes is still a pair that teaches no
# transition, and that is the question.
#
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/c50-change-%j.out
#SBATCH --error=logs/c50-change-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
python3 - <<'PY'
import json, collections
d = json.load(open('data/video/cholect50/raw/samples.json'))
samples = d['samples'] if isinstance(d, dict) and 'samples' in d else d

def triplets_of(s):
    for k, v in s.items():
        if not isinstance(v, dict):
            continue
        cls = v.get('classifications') or v.get('detections')
        if isinstance(cls, list):
            got = [str(c.get('label')) for c in cls
                   if c.get('label') and ',' in str(c.get('label'))]
            if got:
                return got
    return []

# (video, instrument, target) -> ordered list of verbs, by frame_number
seq = collections.defaultdict(list)
rows = 0
per_video_frames = collections.Counter()
for s in samples:
    vid = s.get('video_id')
    fn = s.get('frame_number')
    per_video_frames[vid] += 1
    for t in triplets_of(s):
        parts = [x.strip() for x in t.split(',')]
        if len(parts) != 3:
            continue
        i, v, g = parts
        rows += 1
        seq[(vid, i, g)].append((fn, v))

single = samev = change = 0
changed_examples = []
for k, lst in seq.items():
    lst.sort()
    verbs = [v for _, v in lst]
    if len(verbs) == 1:
        single += 1
    elif len(set(verbs)) == 1:
        samev += 1
    else:
        change += 1
        if len(changed_examples) < 10:
            # the distinct verb sequence, collapsed
            collapsed = []
            for v in verbs:
                if not collapsed or collapsed[-1] != v:
                    collapsed.append(v)
            changed_examples.append((k, collapsed[:6]))
n = len(seq) or 1
print('CholecT50, the THIRD number')
print('  videos                        %d' % len(per_video_frames))
print('  triplet rows                  %d' % rows)
print('  (video, instrument, target) PAIRS  %d' % len(seq))
print('   seen in one frame only       %6d  (%.1f%%)' % (single, 100.0*single/n))
print('   many frames, SAME verb       %6d  (%.1f%%)' % (samev, 100.0*samev/n))
print('   VERB CHANGES                 %6d  (%.1f%%)' % (change, 100.0*change/n))
print()
print('  A PAIR HERE IS CATEGORY LEVEL. There is no instrument instance id,')
print('  so two graspers in one frame collapse into one pair. This number is')
print('  NOT directly comparable with VidVRD, whose pair is (tid, tid).')
print()
print('  example changing pairs, verb sequence collapsed:')
for k, cs in changed_examples:
    print('    %-34s %s' % ('%s / %s -> %s' % k, ' -> '.join(cs)))
PY
