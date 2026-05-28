#!/usr/bin/env bash
# sh/download_videonet.sh — Extract frames from manually-placed VideoNet MP4s.
#
# VideoNet training-data parquets give only YouTube id + timespan (clips are
# YouTube-sourced, NOT hosted on HF; the videos/ UUID folder is benchmark-only).
# So we sidestep both: you drop the chosen clips into videos/ by hand, this
# script just decimates them to frames.
#
# Source:   https://huggingface.co/datasets/raivn/VideoNet
# Tools:    ffmpeg, ffprobe.
#
# Usage
# -----
#   # 1. Drop clips:  data/video/videonet/videos/<stem>.mp4
#   # 2. Extract:
#   FPS=5 bash sh/download_videonet.sh
#
# Env knobs
#   FPS      frame extraction rate (default 5).
#   FORCE=1  re-extract even if a frames dir already exists.
#
# Frame filenames are source-PTS indices (NNNNNN.jpg = source frame N) to match
# puzzle_vidvrd.build_dataset (line ~167, f"{fid:06d}.jpg").

set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FPS="${FPS:-5}"
DATA_DIR="${PROJECT_DIR}/data/video/videonet"
VIDS_DIR="${DATA_DIR}/videos"
FRAMES_DIR="${DATA_DIR}/frames_${FPS}fps"

mkdir -p "${VIDS_DIR}" "${FRAMES_DIR}"

if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
    echo "ERROR: ffmpeg/ffprobe missing. Install or load module." >&2
    exit 4
fi

shopt -s nullglob
MP4S=("${VIDS_DIR}"/*.mp4)
shopt -u nullglob
if (( ${#MP4S[@]} == 0 )); then
    echo "ERROR: no MP4s in ${VIDS_DIR}. Drop pen_spinning clips there first." >&2
    exit 1
fi

echo "[extract] ${#MP4S[@]} clip(s) → frames at ~${FPS}fps (source-PTS names)"
for MP4 in "${MP4S[@]}"; do
    STEM="$(basename -- "${MP4}" .mp4)"
    OUT="${FRAMES_DIR}/${STEM}"
    if [[ -d "${OUT}" && -n "$(ls -A "${OUT}" 2>/dev/null)" && "${FORCE:-0}" != "1" ]]; then
        echo "  [skip] ${STEM} frames present"
        continue
    fi
    SRC_FPS_RATIONAL=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=r_frame_rate -of csv=p=0 "${MP4}")
    SRC_FPS=$(python3 -c "n,d=map(int,'${SRC_FPS_RATIONAL}'.split('/'));print(round(n/d))" \
        2>/dev/null || echo 30)
    SKIP=$(( SRC_FPS / FPS ))
    (( SKIP < 1 )) && SKIP=1
    rm -rf "${OUT}"; mkdir -p "${OUT}"
    echo "  [extract] ${STEM}  src_fps=${SRC_FPS}  skip=${SKIP}"
    if ffmpeg -i "${MP4}" \
              -vf "select='not(mod(n,${SKIP}))'" \
              -vsync vfr -frame_pts true -q:v 2 \
              "${OUT}/%06d.jpg" -loglevel error; then
        N=$(ls "${OUT}" | wc -l)
        echo "  [done] ${STEM} → ${N} frames"
    else
        echo "  [fail] ${STEM} ffmpeg returned non-zero" >&2
        rm -rf "${OUT}"
    fi
done

echo
echo "[ok] frames: ${FRAMES_DIR}"
echo "     Next: bash sh/setup_venv_detect.sh   (one-time sidecar venv)"
