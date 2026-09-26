#!/usr/bin/env bash
#
# unpack_easg.sbatch — open Ego4D-EASG on a compute node.
#
#   sbatch --job-name=easg-unpack sh/unpack_easg.sbatch
#
# WHY A JOB. 18,828 JPEG members out of a 6.1 GB archive is exactly the shape
# of work that was run on the login node on 2026-09-23 and had to be killed.
#
# WHY A MODULE. The system python is 3.6 and its zipfile REFUSES this archive
# with "zipfiles that span multiple disks are not supported". The archive is
# sound: it is zip64 and its end-of-central-directory locator carries a total
# disk count of 0 where 3.6 demands exactly 1. The stock `unzip` fails too,
# with "start of central directory not found", which is the same defect in an
# older reader. Python 3.12.1 opens it and lists all 26,399 members.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
SRC="${ROOT}/data/video/easg/raw/EASG.zip"
DEST="${ROOT}/data/video/easg/ext"

module load python/3.12.1
mkdir -p "${DEST}"

python3 - "${SRC}" "${DEST}" <<'PY'
import sys, zipfile, os
src, dest = sys.argv[1], sys.argv[2]
z = zipfile.ZipFile(src)
names = z.namelist()
print("members: %d" % len(names), flush=True)
done = 0
for n in names:
    z.extract(n, dest)
    done += 1
    if done % 2000 == 0:
        print("  %d/%d" % (done, len(names)), flush=True)
print("extracted %d" % done)
jpg = sum(1 for n in names if n.lower().endswith(".jpg"))
print("jpg members: %d" % jpg)
PY

echo "on disk: $(find "${DEST}" -name '*.jpg' | wc -l) jpg"
du -sh "${DEST}"
