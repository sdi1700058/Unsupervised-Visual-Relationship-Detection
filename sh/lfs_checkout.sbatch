#!/usr/bin/env bash
#
# lfs_checkout.sbatch — materialise a git-lfs clone whose pointers never
# resolved.
#
# WHY THIS IS NOT A RE-DOWNLOAD. Measured 2026-09-22: CholecT50's clone holds
# 60 GB under data/video/cholect50/hf/.git and 100,863 files named `.png` that
# are each a 131-byte pointer reading
# `version https://git-lfs.github.com/spec/v1`. The BYTES ARE ALREADY HERE.
# The smudge filter never ran, because git-lfs is not on the login node's path,
# so `git lfs checkout` writes the pixels from the local object store and
# fetches nothing.
#
# The board called that dataset LOCKABLE on the strength of those pointers,
# which is the defect `tools/planner/tests/test_lock_board.py` now pins.
#
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/lfsco-%x-%j.out
#SBATCH --error=logs/lfsco-%x-%j.err
set -uo pipefail
DS="${1:?usage: sbatch sh/lfs_checkout.sbatch <dataset>}"
ROOT="${SLURM_SUBMIT_DIR:?}"
DEST="${ROOT}/data/video/${DS}/hf"
[[ -d "${DEST}/.git" ]] || { echo "no clone at ${DEST}"; exit 2; }

ml load system git-lfs 2>/dev/null || ml load git-lfs 2>/dev/null || true
command -v git-lfs >/dev/null || { echo "git-lfs unavailable on this node"; exit 3; }
git lfs version

export HF_HOME="${ROOT}/data/.hf_home"
export XDG_CACHE_HOME="${ROOT}/data/.cache"
cd "${DEST}" || exit 4

echo "before: real png >2k = $(find . -path ./.git -prune -o -name '*.png' -size +2k -print 2>/dev/null | wc -l)"
git lfs install --local
git lfs checkout 2>&1 | tail -5
rc=$?
echo "checkout exit ${rc}"
# A checkout that finds nothing locally still needs the objects. Try a pull
# only if the checkout left pointers behind.
left=$(find . -path ./.git -prune -o -name '*.png' -size -2k -print 2>/dev/null | wc -l)
if [[ "${left}" -gt 0 ]]; then
    echo "${left} pointers remain, running git lfs pull"
    git lfs pull 2>&1 | tail -5
fi

real=$(find . -path ./.git -prune -o -name '*.png' -size +2k -print 2>/dev/null | wc -l)
ptr=$(find . -path ./.git -prune -o -name '*.png' -size -2k -print 2>/dev/null | wc -l)
echo "after : real png = ${real}, pointers left = ${ptr}"
{
  echo "dataset=${DS}"
  echo "measured_on=$(date +%Y-%m-%d)"
  echo "real_frames=${real}"
  echo "pointers_left=${ptr}"
} > "${ROOT}/eval/datasets/routes/${DS}-checkout.fetched"
[[ "${real}" -gt 0 ]] || { echo "checkout produced NO pixels"; exit 6; }
