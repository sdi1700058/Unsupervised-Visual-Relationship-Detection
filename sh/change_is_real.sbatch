#!/usr/bin/env bash
#
# change_is_real.sbatch — is a predicate change real motion, or quantisation?
#
# TWO ADVERSARIES NAMED THIS AS THE CHECK THEY COULD NOT REACH, on 2026-09-22.
# The whole ranking rests on how many object pairs ever CHANGE their relation,
# and 59.4 per cent of VidVRD's changing pairs only flip a DIRECTION TOKEN:
# stand_left becomes stand_right. That is either a dog walking past a frisbee
# or a static box flickering across a quantisation boundary, and the two look
# identical in the label.
#
# THE TEST IS FALSIFIABLE. VidVRD's direction tokens are defined on the
# relative position of the two boxes. If stand_left becoming stand_right is
# real, the SIGN of (subject centre x minus object centre x) must flip between
# the two segments. If it does not flip, the label moved and the objects did
# not.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/changereal-%j.out
#SBATCH --error=logs/changereal-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
python3 - <<'PY'
import json, glob, collections, math

PERSON = {'person', 'adult', 'child', 'baby', 'man', 'woman', 'people'}
LR = {'left': -1, 'right': +1}
FB = {'front': +1, 'behind': -1}


def centre(b):
    return ((b['xmin'] + b['xmax']) / 2.0, (b['ymin'] + b['ymax']) / 2.0)


def diag(b):
    return math.hypot(b['xmax'] - b['xmin'], b['ymax'] - b['ymin']) or 1.0


def rel_at(traj, fid, stid, otid):
    """Relative offset of subject from object at one frame, and a scale."""
    if fid < 0 or fid >= len(traj):
        return None
    s = o = None
    for box in traj[fid]:
        if box['tid'] == stid:
            s = box['bbox']
        elif box['tid'] == otid:
            o = box['bbox']
    if s is None or o is None:
        return None
    sc, oc = centre(s), centre(o)
    return (sc[0] - oc[0], sc[1] - oc[1], (diag(s) + diag(o)) / 2.0)


files = sorted(glob.glob('data/video/vidvrd/annotations/*/*.json'))
pairs = 0
dir_flip_pairs = 0
sign_flipped = 0
sign_held = 0
no_geometry = 0
disp = []
held_examples = []
flip_examples = []

for f in files:
    d = json.load(open(f))
    t2c = {t['tid']: t['category'] for t in d.get('subject/objects', [])}
    traj = d.get('trajectories', [])
    grouped = collections.defaultdict(list)
    for r in d.get('relation_instances', []):
        s = t2c.get(r['subject_tid'], '?')
        o = t2c.get(r['object_tid'], '?')
        if s in PERSON or o in PERSON:
            continue
        grouped[(r['subject_tid'], r['object_tid'])].append(r)
    for (stid, otid), rs in grouped.items():
        preds = {r['predicate'] for r in rs}
        if len(preds) < 2:
            continue
        pairs += 1
        # Only the direction-token pairs can be tested this way.
        roots = {p.rsplit('_', 1)[0] for p in preds if '_' in p}
        tails = {p.rsplit('_', 1)[1] for p in preds if '_' in p}
        if len(roots) != 1 or not (tails <= set(LR) or tails <= set(FB)):
            continue
        axis, table = (0, LR) if tails <= set(LR) else (1, FB)
        dir_flip_pairs += 1
        rs.sort(key=lambda r: r['begin_fid'])
        # Compare the FIRST segment against the first later segment whose
        # direction token differs.
        first = rs[0]
        ft = first['predicate'].rsplit('_', 1)[1]
        other = None
        for r in rs[1:]:
            if r['predicate'].rsplit('_', 1)[1] != ft:
                other = r
                break
        if other is None:
            continue
        a = rel_at(traj, (first['begin_fid'] + first['end_fid']) // 2, stid, otid)
        b = rel_at(traj, (other['begin_fid'] + other['end_fid']) // 2, stid, otid)
        if a is None or b is None:
            no_geometry += 1
            continue
        va, vb = a[axis], b[axis]
        scale = (a[2] + b[2]) / 2.0
        disp.append(abs(vb - va) / scale)
        if (va > 0) != (vb > 0):
            sign_flipped += 1
            if len(flip_examples) < 4:
                flip_examples.append((t2c[stid], first['predicate'],
                                      other['predicate'], t2c[otid],
                                      round(va, 1), round(vb, 1)))
        else:
            sign_held += 1
            if len(held_examples) < 6:
                held_examples.append((t2c[stid], first['predicate'],
                                      other['predicate'], t2c[otid],
                                      round(va, 1), round(vb, 1)))

disp.sort()


def q(p):
    return disp[int(p * (len(disp) - 1))] if disp else float('nan')


print('VidVRD: IS A DIRECTION FLIP REAL MOTION?')
print('  object-object pairs with more than one predicate   %d' % pairs)
print('  of those, pure DIRECTION-TOKEN pairs               %d' % dir_flip_pairs)
print('  testable with geometry at both segment midpoints   %d'
      % (sign_flipped + sign_held))
print('  geometry missing at a midpoint                     %d' % no_geometry)
print()
print('  THE SIGN FLIPPED, so the objects really crossed    %d  (%.1f%%)'
      % (sign_flipped, 100.0 * sign_flipped / max(1, sign_flipped + sign_held)))
print('  THE SIGN HELD, so the LABEL moved and they did not %d  (%.1f%%)'
      % (sign_held, 100.0 * sign_held / max(1, sign_flipped + sign_held)))
print()
print('  relative displacement between the two segments,')
print('  in units of mean box diagonal:')
for p in (0.1, 0.25, 0.5, 0.75, 0.9):
    print('     p%02d  %.3f' % (int(p * 100), q(p)))
print()
print('  SIGN HELD examples (subject, from, to, object, offset_a, offset_b):')
for e in held_examples:
    print('     %s' % (e,))
print('  SIGN FLIPPED examples:')
for e in flip_examples:
    print('     %s' % (e,))
PY
