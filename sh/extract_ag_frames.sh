#!/usr/bin/env bash
# extract_ag_frames.sh — ffmpeg extract of ActionGenome-annotated frames.
#
# Wraps upstream `data/video/actiongenome/ag_repo/tools/dump_frames.py`.
# Keeps only the frames listed in `annotations/frame_list.txt` (post-extract
# prune handled inside dump_frames.py when --all_frames is omitted).
#
# Per V12: this script is sbatch-safe — activates the venv directly,
# never calls `module purge`. ffmpeg is sourced from the system PATH;
# if absent on the compute node, we fall back to `module load ffmpeg`
# after an explicit modules-init.
#
# Usage (sbatch):
#   sbatch sh/extract_ag_frames.sh
# Usage (interactive smoke, first 5 vids only — set MAX before invocation):
#   MAX=5 bash sh/extract_ag_frames.sh
#
#SBATCH --job-name=fosae-ag-extract
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -eo pipefail
# In sbatch, BASH_SOURCE points at a spool copy of the script (e.g.
# /var/spool/slurmd/job.../slurm_script) — useless for resolving the project
# root. Use SLURM_SUBMIT_DIR when present, else fall back to BASH_SOURCE for
# interactive invocation.  (V12 / B13 fix.)
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

# Prefer sherlock_env.sh (loads modules + venv consistently). Fallback to bare
# venv activate so out-of-tree invocations still work.
if [[ -f "${PROJECT_DIR}/sh/sherlock_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/sh/sherlock_env.sh"
else
    source "${PROJECT_DIR}/venv/bin/activate"
fi

# ensure ffmpeg available; tolerate either system PATH or Sherlock module
if ! command -v ffmpeg >/dev/null 2>&1; then
    # initialise the modules command for non-login shells
    [[ -r /etc/profile.d/modules.sh ]] && . /etc/profile.d/modules.sh
    module load ffmpeg 2>/dev/null || {
        echo "ERROR: ffmpeg not in PATH and module load failed" >&2
        exit 4
    }
fi
echo "[ag-extract] ffmpeg: $(which ffmpeg)"

AG_ROOT="${PROJECT_DIR}/data/video/actiongenome"
VIDEO_DIR="${AG_ROOT}/charades_videos/Charades_v1_480"
ANN_DIR="${AG_ROOT}/annotations"
FRAME_DIR="${AG_ROOT}/frames"
mkdir -p "${FRAME_DIR}" logs

# Smoke mode: trim frame_list.txt to first MAX videos
ANN_DIR_RUN="${ANN_DIR}"
if [[ -n "${MAX:-}" ]]; then
    TRIM_DIR="$(mktemp -d -t ag-smoke-XXXXXX)"
    ln -s "${ANN_DIR}/object_bbox_and_relationship.pkl"  "${TRIM_DIR}/"
    ln -s "${ANN_DIR}/person_bbox.pkl"                   "${TRIM_DIR}/"
    awk -F/ -v M="${MAX}" '{ if (!(($1) in seen)) { seen[$1]=1; if (++n>M) exit } print }' \
        "${ANN_DIR}/frame_list.txt" > "${TRIM_DIR}/frame_list.txt"
    ANN_DIR_RUN="${TRIM_DIR}"
    echo "[ag-extract] MAX=${MAX} smoke mode -> $(wc -l < ${TRIM_DIR}/frame_list.txt) frames across ${MAX} videos"
fi

python3 "${AG_ROOT}/ag_repo/tools/dump_frames.py" \
    --video_dir "${VIDEO_DIR}" \
    --frame_dir "${FRAME_DIR}" \
    --annotation_dir "${ANN_DIR_RUN}"

DIR_COUNT=$(ls "${FRAME_DIR}" 2>/dev/null | wc -l)
FILE_COUNT=$(find "${FRAME_DIR}" -mindepth 2 -name '*.png' | wc -l)
echo "[ag-extract] DONE — videos=${DIR_COUNT}, frames=${FILE_COUNT}"
