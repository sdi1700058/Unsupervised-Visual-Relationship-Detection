#!/usr/bin/env bash
#
# count_cholect50_relations.sbatch — the relation SHAPE for CholecT50.
#
# WHY. The object-to-object split destroyed MOMA-LRG (2 of 985,243) and Home
# Action Genome (0 of 583,481) on 2026-09-22, and both failed for the same
# structural reason: every relation hung off ONE node, a person. CholecT50's
# triplet is (instrument, verb, target), so neither end is a person and the
# naive split answers 100 per cent.
#
# THAT ANSWER WOULD BE MISLEADING AND THIS JOB EXISTS TO SAY SO. The instrument
# is the surgeon's proxy, so a star graph around the instrument is the same
# shape as a star graph around a person. What decides it is whether any
# relation joins two ANATOMICAL targets, and whether more than one instrument
# ever appears in a frame.
#
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/c50-rel-%j.out
#SBATCH --error=logs/c50-rel-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
python3 - <<'PY'
import json, collections
p = 'data/video/cholect50/raw/samples.json'
d = json.load(open(p))
samples = d['samples'] if isinstance(d, dict) and 'samples' in d else d
print('samples: %d' % len(samples))
print('keys of one sample: %s' % sorted(samples[0].keys())[:30])

trip = collections.Counter()
inst = collections.Counter(); verb = collections.Counter(); targ = collections.Counter()
per_frame = collections.Counter()
instr_per_frame = collections.Counter()
targ_per_frame = collections.Counter()
frames_with_two_instruments = 0
frames_with_two_targets = 0
pairs = collections.Counter()
fields = collections.Counter()

def triplets_of(s):
    """Pull the triplet list out of whatever field carries it."""
    for k, v in s.items():
        if not isinstance(v, dict):
            continue
        cls = v.get('classifications') or v.get('detections')
        if isinstance(cls, list):
            got = []
            for c in cls:
                lab = c.get('label')
                if lab and ',' in str(lab):
                    got.append(str(lab))
            if got:
                fields[k] += 1
                return got
    return []

for s in samples:
    ts = triplets_of(s)
    per_frame[len(ts)] += 1
    fi = set(); ft = set()
    for t in ts:
        parts = [x.strip() for x in t.split(',')]
        if len(parts) != 3:
            continue
        i, v, g = parts
        trip[t] += 1; inst[i] += 1; verb[v] += 1; targ[g] += 1
        pairs[(i, g)] += 1
        fi.add(i); ft.add(g)
    instr_per_frame[len(fi)] += 1
    targ_per_frame[len(ft)] += 1
    if len(fi) > 1: frames_with_two_instruments += 1
    if len(ft) > 1: frames_with_two_targets += 1

print()
print('field carrying the triplet: %s' % dict(fields))
print('distinct triplets   %d' % len(trip))
print('instruments %d: %s' % (len(inst), dict(inst.most_common())))
print('verbs       %d: %s' % (len(verb), dict(verb.most_common())))
print('targets     %d: %s' % (len(targ), dict(targ.most_common())))
print()
print('triplets per frame  %s' % dict(sorted(per_frame.items())))
print('instruments/frame   %s' % dict(sorted(instr_per_frame.items())))
print('targets/frame       %s' % dict(sorted(targ_per_frame.items())))
print('frames with >1 instrument %d' % frames_with_two_instruments)
print('frames with >1 target     %d' % frames_with_two_targets)
print()
print('THE SHAPE QUESTION: is any relation anatomy-to-anatomy?')
print('  Every triplet is (instrument, verb, target) by construction, so the')
print('  answer is NO by the schema. distinct (instrument,target) pairs: %d'
      % len(pairs))
for k, v in pairs.most_common(20):
    print('    %-22s -> %-22s %8d' % (k[0], k[1], v))
PY
