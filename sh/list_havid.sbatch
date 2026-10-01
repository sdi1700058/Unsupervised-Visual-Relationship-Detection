#!/usr/bin/env bash
#
# list_havid.sbatch — what is inside each HA-ViD zip, from the central directories
#
#   sbatch --job-name=havid-list sh/list_havid.sbatch
#
# WHY. The author uploaded HA-ViD on 2026-09-29. ObjectDetection.zip alone is
# 117,591,123,942 bytes, so it is not unpacked blind: this reads only each
# archive's central directory and reports members per folder and per file
# type, plus every member that is not an image or a video, which is where the
# annotations live. Nothing is extracted.
#
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load python/3.9.0
python3 - "${ROOT}/data/video/ha_vid/raw" <<'PY'
import collections, os, sys, zipfile
raw = sys.argv[1]
MEDIA = (".jpg", ".jpeg", ".png", ".mp4", ".avi", ".mkv", ".npy")
for name in sorted(os.listdir(raw)):
    if not name.endswith(".zip"):
        continue
    z = zipfile.ZipFile(os.path.join(raw, name))
    infos = [i for i in z.infolist() if not i.is_dir()]
    print("=== %s: %d files, %d bytes uncompressed"
          % (name, len(infos), sum(i.file_size for i in infos)), flush=True)
    by_dir = collections.Counter()
    by_ext = collections.Counter()
    other = []
    for i in infos:
        parts = i.filename.split("/")
        by_dir["/".join(parts[:3])] += 1
        ext = os.path.splitext(i.filename)[1].lower()
        by_ext[ext] += 1
        if ext not in MEDIA:
            other.append((i.filename, i.file_size))
    print("  by extension: %s" % dict(by_ext.most_common()))
    print("  folders, first three levels:")
    for d, c in sorted(by_dir.items())[:80]:
        print("    %8d  %s" % (c, d))
    print("  non-media members: %d, first 60:" % len(other))
    for f, size in other[:60]:
        print("    %12d  %s" % (size, f))
PY
