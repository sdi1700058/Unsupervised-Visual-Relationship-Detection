#!/usr/bin/env bash
#
# unpack_ssv2.sbatch — unpack SSv2's 220,847 videos, the frames under Something-Else
#
#   sbatch --job-name=ssv2-unpack sh/unpack_ssv2.sbatch
#
# WHY. The lock board reads Something-Else as ORACLE ARM only because SSv2
# sits in twenty split gzip parts it cannot index. ssv2-list, Slurm 45566771,
# listed 220,847 webm videos through tar -tz on 2026-09-27, which meets the
# author's lock rule by count. An adversary found on 2026-10-01 that the tool
# and the rule disagree here, so this closes the gap by measurement: the
# videos land in data/video/ssv2/ext, where the board can count them.
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
