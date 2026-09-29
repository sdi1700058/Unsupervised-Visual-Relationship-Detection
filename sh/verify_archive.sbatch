#!/usr/bin/env bash
#
# verify_archive.sbatch — is a download, and its unpacked copy, correct?
#
#   sbatch --job-name=verify-easg sh/verify_archive.sbatch \
#       data/video/easg/raw/EASG.zip data/video/easg/ext --images
#
# Re-reads every member against its stored CRC, checks the unpacked copy name
# for name and size for size, and with --images decodes every image in full.
# A size that matches Content-Length says the byte count arrived and nothing
# more, which is why this job exists. See workbench/tools/verify_archive.py.
#
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
ZIP="${1:?usage: sbatch sh/verify_archive.sbatch <zip> [unpack root] [--images]}"
shift
ARGS=()
if [ $# -gt 0 ] && [ "$1" != "--images" ]; then
    ARGS+=(--root "${ROOT}/$1")
    shift
fi
ARGS+=("$@")
# PYTHON 3.9, NOT THE 3.6 VENV. Measured 2026-09-29: EASG.zip's ZIP64
# locator records 0 disks where the format expects 1, and Python 3.6's zipfile
# refuses that as "zipfiles that span multiple disks are not supported" while
# 3.9 opens it. That is a quirk of the writer, not corruption.
module load python/3.9.0
module load viz
module load py-pillow/9.3.0_py39
cd "${ROOT}/workbench" || exit 2
PYTHONPATH="..:${PYTHONPATH:-}" python3 \
    -m tools.verify_archive "${ROOT}/${ZIP}" "${ARGS[@]}"
