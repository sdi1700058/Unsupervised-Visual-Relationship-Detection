#!/usr/bin/env bash
#
# unpack_ssv2.sbatch — unpack SSv2's 220,847 videos, the frames under Something-Else
#
#   sbatch --job-name=ssv2-unpack runs/data/unpack_ssv2.sbatch
#
# WHY. runs/data/convert_something_else.sbatch (the C.4 unpack step) reads its webm
# source from data/video/ssv2/ext; this script rebuilds that directory from
# the raw gzip parts after a $SCRATCH purge removes it.
#
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
RAW="${ROOT}/data/video/ssv2/raw"
DEST="${ROOT}/data/video/ssv2/ext"
mkdir -p "${DEST}"
cat "${RAW}"/20bn-something-something-v2-?? | tar -xzf - -C "${DEST}"
status=( "${PIPESTATUS[@]}" )
echo "cat exit ${status[0]}, tar exit ${status[1]}"
echo "webm files: $(find "${DEST}" -name '*.webm' | wc -l)"
echo "bytes: $(du -sb "${DEST}" | cut -f1)"
[ "${status[0]}" -eq 0 ] && [ "${status[1]}" -eq 0 ]
