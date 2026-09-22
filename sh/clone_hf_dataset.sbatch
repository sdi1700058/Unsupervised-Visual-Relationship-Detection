#!/usr/bin/env bash
#
# clone_hf_dataset.sbatch — pull a HuggingFace dataset repo whose payload is
# tens of thousands of loose files rather than a few archives.
#
#   sbatch --job-name=dl-c50-frames sh/clone_hf_dataset.sbatch \
#          Voxel51/cholect50 cholect50
#
# WHY THIS EXISTS BESIDE fetch_dataset.sbatch. That script walks a manifest of
# URLs, which is right for VidOR's twelve archives and wrong for CholecT50's
# 101 shards of 1,000 PNG each. 101,000 separate HTTPS requests is slow, it
# retries badly, and a manifest that long is not reviewable.
#
# `git clone` with git-lfs does the same work in one connection, resumes on
# its own, and verifies each object against its hash, which a bare curl does
# not. git-lfs is a Sherlock module, measured 2026-09-21 by `ml spider`.
#
# THE REPOSITORY IS NOT THE DESTINATION. The clone lands in
# data/video/<name>/hf, which is gitignored, so the private repository never
# sees 58 GB of frames. `git clean -x` would destroy the nested workbench
# repository and is forbidden by CLAUDE.md, so nothing here calls it.
#
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=logs/clone-%x-%j.out
#SBATCH --error=logs/clone-%x-%j.err

set -uo pipefail

REPO="${1:?usage: sbatch sh/clone_hf_dataset.sbatch <org/name> <local-name>}"
DS="${2:?usage: sbatch sh/clone_hf_dataset.sbatch <org/name> <local-name>}"

ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
[[ -d "${ROOT}/eval/datasets/routes" ]] || {
    echo "clone: ${ROOT} is not the repository root." >&2; exit 2; }

DEST="${ROOT}/data/video/${DS}/hf"
say () { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

mkdir -p "$(dirname "${DEST}")" "${ROOT}/logs"

say "loading git-lfs"
ml load system git-lfs 2>/dev/null || ml load git-lfs 2>/dev/null || true
command -v git-lfs >/dev/null || { say "git-lfs unavailable"; exit 3; }
git lfs install --local 2>/dev/null || git lfs install 2>/dev/null || true

# Everything here is cache and must not land in $HOME, which is 15 GB and
# NFS-backed. The cluster policy says so directly.
export GIT_LFS_SKIP_SMUDGE=0
export HF_HOME="${ROOT}/data/.hf_home"
export XDG_CACHE_HOME="${ROOT}/data/.cache"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}"

if [[ -d "${DEST}/.git" ]]; then
    say "resuming ${REPO} in ${DEST}"
    cd "${DEST}" || exit 4
    git fetch --all && git checkout . && git lfs pull
else
    say "cloning ${REPO} into ${DEST}"
    git clone "https://huggingface.co/datasets/${REPO}" "${DEST}" || exit 5
    cd "${DEST}" || exit 4
    git lfs pull
fi

rc=$?
say "git-lfs pull exit ${rc}"

frames=$(find "${DEST}" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.mp4' \) 2>/dev/null | wc -l)
bytes=$(du -sb "${DEST}" 2>/dev/null | cut -f1)
say "frames on disk: ${frames}"
say "bytes on disk : ${bytes}"

{
    echo "dataset=${DS}"
    echo "repo=${REPO}"
    echo "measured_on=$(date +%Y-%m-%d)"
    echo "frames=${frames}"
    echo "bytes=${bytes}"
} > "${ROOT}/eval/datasets/routes/${DS}-frames.fetched"

# THE COUNT IS THE POINT. A clone that finishes with no frames is a failure
# that looks like a success, which is the shape this project has been burned
# by more than once.
[[ "${frames}" -gt 0 ]] || { say "clone produced NO frames"; exit 6; }
exit "${rc}"
