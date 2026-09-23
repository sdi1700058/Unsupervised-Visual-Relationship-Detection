#!/usr/bin/env bash
#
# objects_per_frame.sbatch — how hard does --max-objects 3 actually bite?
#
# AN ADVERSARY NAMED THIS AS THE CHECK IT DID NOT REACH, and said so. It
# measured objects per CLIP from the length of `subject/objects`, which found
# 61.1 per cent of VidOR clips over 3, and warned in its own report that a
# whole-clip count OVERSTATES the pressure: ten objects across a clip may be
# two or three visible at once.
#
# The pipeline binds a fixed number of slots, `--max-objects` default 3 in
# five entry points, so what matters is the CONCURRENT count in one frame.
# This walks every trajectory and counts, per frame, how many objects are
# actually present.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/objperframe-%j.out
#SBATCH --error=logs/objperframe-%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}" || exit 2
python3 - <<'PY'
import json, glob, collections

def survey(name, files):
    per_frame = collections.Counter()      # count -> frames
    per_clip_max = collections.Counter()   # clip max concurrent -> clips
    clips_over = 0
    clips = 0
    frames = 0
    traj_len = collections.Counter()
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        clips += 1
        traj_len[len(d.get('subject/objects', []))] += 1
        # trajectories: one list per frame of {tid, bbox}
        traj = d.get('trajectories') or []
        mx = 0
        for fr in traj:
            frames += 1
            n = len(fr)
            per_frame[n] += 1
            if n > mx:
                mx = n
        per_clip_max[mx] += 1
        if mx > 3:
            clips_over += 1
    tot_frames = sum(per_frame.values()) or 1
    over3 = sum(v for k, v in per_frame.items() if k > 3)
    print('%s' % name)
    print('  clips                         %d' % clips)
    print('  annotated frames              %d' % tot_frames)
    print('  FRAMES with more than 3 objects present  %d  (%.1f%%)'
          % (over3, 100.0*over3/tot_frames))
    print('  clips whose MAX concurrent exceeds 3     %d  (%.1f%%)'
          % (clips_over, 100.0*clips_over/(clips or 1)))
    print('  objects present per frame, distribution:')
    for k in sorted(per_frame):
        print('     %2d objects  %8d frames  (%.1f%%)'
              % (k, per_frame[k], 100.0*per_frame[k]/tot_frames))
    print('  objects per CLIP (the whole-clip count the adversary used):')
    for k in sorted(traj_len):
        print('     %2d objects  %6d clips' % (k, traj_len[k]))
    print()

survey('VidVRD', sorted(glob.glob('data/video/vidvrd/annotations/*/*.json')))
survey('VidOR', sorted(glob.glob('data/video/vidor/annotations/**/*.json',
                                 recursive=True)))
PY
