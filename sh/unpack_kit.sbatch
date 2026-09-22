#!/usr/bin/env bash
#
# unpack_kit.sbatch — unzip KIT's derived data, then measure it.
#
# WHY A JOB AND NOT A SHELL. The spatial relations are ONE JSON PER FRAME, so
# the zip holds roughly 220,000 small files. A login-node unzip had written 53
# of 540 takes in eight minutes and was hammering the Lustre metadata server,
# which /etc/claude-code/CLAUDE.md forbids by name.
#
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/kit-unpack-%j.out
#SBATCH --error=logs/kit-unpack-%j.err
set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
cd "${ROOT}" || exit 2
RAW="data/video/kit_bimanual/raw"
EXT="data/video/kit_bimanual/ext"
mkdir -p "${EXT}"
for z in bimacs_derived_data_spatial_relations bimacs_derived_data_3d_objects \
         bimacs_derived_data_2d_objects bimacs_rgbd_data_ground_truth; do
    [[ -r "${RAW}/${z}.zip" ]] || { echo "missing ${z}.zip"; continue; }
    echo "== ${z} =="
    unzip -q -o "${RAW}/${z}.zip" -d "${EXT}/" && echo "   ok"
    echo "   files now: $(find "${EXT}" -type f | wc -l)"
done
echo "TOTAL files: $(find "${EXT}" -type f | wc -l)"
echo "takes with spatial_relations: $(find "${EXT}" -type d -name spatial_relations | wc -l)"
echo
echo "===================== MEASURE ====================="
python3 - <<'PY'
import json, os, glob, collections
ROOT = 'data/video/kit_bimanual/ext/bimacs_derived_data'
rel_files = glob.glob(os.path.join(ROOT, '*', '*', '*', 'spatial_relations', '*.json'))
obj_files = glob.glob(os.path.join(ROOT, '*', '*', '*', '3d_objects', '*.json'))
print('spatial_relations frame files: %d' % len(rel_files))
print('3d_objects frame files       : %d' % len(obj_files))

def take_of(p):
    return os.sep.join(p.split(os.sep)[:-2])

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
        names_by_take[t] = {i: str(o.get('class_name') or o.get('name') or '?')
                            for i, o in enumerate(d)}
print('takes with an object name table: %d' % len(names_by_take))
for t, n in list(names_by_take.items())[:2]:
    print('  %s -> %s' % (os.sep.join(t.split(os.sep)[-3:]), n))

HANDISH = ('hand',)
rel = collections.Counter(); classes = collections.Counter()
tot = oo = hand = 0
pairseq = collections.defaultdict(list)
takes = set()

for p in sorted(rel_files):
    t = take_of(p); takes.add(t)
    names = names_by_take.get(t, {})
    try:
        d = json.load(open(p))
    except Exception:
        continue
    per_pair = collections.defaultdict(set)
    for r in d:
        s = r.get('subject_index'); o = r.get('object_index')
        n = r.get('relation_name')
        tot += 1; rel[n] += 1
        sn = names.get(s, str(s)).lower(); on = names.get(o, str(o)).lower()
        classes[sn] += 1
        if any(h in sn for h in HANDISH) or any(h in on for h in HANDISH):
            hand += 1
        else:
            oo += 1
            per_pair[(t, s, o)].add(n)
    for k, v in per_pair.items():
        pairseq[k].append(frozenset(v))

single = samep = change = 0
for seq in pairseq.values():
    if len(seq) == 1: single += 1
    elif len(set(seq)) == 1: samep += 1
    else: change += 1
n = len(pairseq) or 1
print()
print('KIT Bimanual, measured')
print('  relation rows (per frame)     %d' % tot)
print('  involves a HAND               %d  (%.1f%%)' % (hand, 100.0*hand/tot if tot else 0))
print('  OBJECT to OBJECT              %d  (%.1f%%)' % (oo, 100.0*oo/tot if tot else 0))
print('  distinct relation names       %d' % len(rel))
print('  takes                         %d' % len(takes))
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
print('  object classes seen')
for k, v in classes.most_common(25):
    print('    %-24s %10d' % (k, v))
PY
