#!/usr/bin/env bash
# sh/download_vidvrd.sh — Download ImageNet-VidVRD and extract frames.
#
# Usage:
#   bash sh/download_vidvrd.sh [--annotations-only] [--force]
#   FPS=3 bash sh/download_vidvrd.sh          # extract at 3fps → data/vidvrd/frames_3fps/ (recommended)
#
# Skips downloads/extraction if already complete. Use --force to re-download.
# Requires: wget, ffmpeg, unzip
# Dataset: https://huggingface.co/datasets/shangxd/imagenet-vidvrd

set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data/video/vidvrd"
RAW_DIR="${DATA_DIR}/raw"
ANN_DIR="${DATA_DIR}/annotations"
FPS="${FPS:-1}"
if (( FPS < 1 )); then
    echo "ERROR: FPS must be >= 1. Got FPS=${FPS}." >&2
    exit 2
fi
if (( FPS > 30 )); then
    echo "WARN: FPS=${FPS} > typical source fps (25-30). Many duplicate frames expected." >&2
fi

# Parallelism + staging area. On Sherlock the Lustre/GPFS $SCRATCH is slow at
# many-small-file create(). Stage to fast local NVMe ($L_SCRATCH or $TMPDIR),
# then mv the per-video dir into FRAMES_DIR at the end.
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || echo 4)}}"
STAGE_BASE="${STAGE_DIR:-${L_SCRATCH:-${TMPDIR:-/tmp}}}"
STAGE_DIR="${STAGE_BASE}/vidvrd_extract_$$"
mkdir -p "${STAGE_DIR}"
trap 'rm -rf "${STAGE_DIR}"' EXIT
if [[ "${FPS}" == "1" ]]; then
    FRAMES_DIR="${DATA_DIR}/frames"
else
    FRAMES_DIR="${DATA_DIR}/frames_${FPS}fps"
fi

ANNOTATIONS_ONLY="${1:-}"
FORCE="${2:-}"

mkdir -p "${RAW_DIR}" "${ANN_DIR}" "${FRAMES_DIR}"

HF_BASE="https://huggingface.co/datasets/shangxd/imagenet-vidvrd/resolve/main"

# ── Check if annotations already downloaded ────────────────────────────────
ANN_COUNT=$(find "${ANN_DIR}" -name "*.json" 2>/dev/null | wc -l)
if [[ ${ANN_COUNT} -gt 0 && "${FORCE}" != "--force" ]]; then
    echo "[1] Annotations already present (${ANN_COUNT} JSONs). Skipping."
else
    echo "[1] Downloading annotations..."
    [[ "${FORCE}" == "--force" ]] && rm -f "${RAW_DIR}/vidvrd-annotations.zip"
    wget -q -c "${HF_BASE}/vidvrd-annotations.zip" -O "${RAW_DIR}/vidvrd-annotations.zip"
    unzip -q -o "${RAW_DIR}/vidvrd-annotations.zip" -d "${ANN_DIR}"
    mv "${ANN_DIR}/vidvrd-dataset/train" "${ANN_DIR}/" 2>/dev/null || true
    mv "${ANN_DIR}/vidvrd-dataset/test" "${ANN_DIR}/" 2>/dev/null || true
    rm -rf "${ANN_DIR}/vidvrd-dataset"
    ANN_COUNT=$(find "${ANN_DIR}" -name "*.json" 2>/dev/null | wc -l)
    echo "  ${ANN_COUNT} JSON files"
fi

[[ "${ANNOTATIONS_ONLY}" == "--annotations-only" ]] && { echo "Done."; exit 0; }

# ── Check if frames already extracted ──────────────────────────────────────
VIDEO_COUNT=$(find "${FRAMES_DIR}" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)
if [[ ${VIDEO_COUNT} -eq 1000 && "${FORCE}" != "--force" ]]; then
    echo "[2] Frames already extracted (${VIDEO_COUNT} videos). Done."
    exit 0
fi

# ── Check if ffmpeg is available ──────────────────────────────────────────
if ! command -v ffmpeg &> /dev/null; then
    echo "[2] ERROR: ffmpeg not found. Install with: conda install ffmpeg or apt-get install ffmpeg"
    exit 1
fi

# ── Download both parts first ──────────────────────────────────────────────
echo "[2] Downloading/checking video zips..."
for PART in part1 part2; do
    ZIP="${RAW_DIR}/vidvrd-videos-${PART}.zip"
    if [[ -f "${ZIP}" && "${FORCE}" != "--force" ]]; then
        echo "  ${PART} zip already present."
    else
        echo "  Downloading vidvrd-videos-${PART}.zip (~2-3 GB)..."
        [[ "${FORCE}" == "--force" ]] && rm -f "${ZIP}"
        wget -q -c "${HF_BASE}/vidvrd-videos-${PART}.zip" -O "${ZIP}"
    fi
done

# ── Extract both parts ─────────────────────────────────────────────────────
echo "[2] Extracting video zips..."
TMPVID="${RAW_DIR}/videos_tmp"
mkdir -p "${TMPVID}"
for PART in part1 part2; do
    ZIP="${RAW_DIR}/vidvrd-videos-${PART}.zip"
    if [[ -f "${ZIP}" ]]; then
        echo "  Extracting ${PART}..."
        unzip -q -o "${ZIP}" -d "${TMPVID}"
    fi
done

# ── Extract frames from all videos ─────────────────────────────────────────
FAILED_LOG="${RAW_DIR}/failed_videos.txt"
> "${FAILED_LOG}"

echo "[2] Extracting frames at ${FPS}fps using ${JOBS} parallel ffmpeg jobs..."
echo "    Stage dir: ${STAGE_DIR}"
echo "    Final dir: ${FRAMES_DIR}"

extract_one() {
    local VID="$1"
    local VID_ID SPLIT_DIR OUT STAGE_OUT SKIP
    VID_ID="$(basename -- "${VID}" .mp4)"

    if [[ -f "${ANN_DIR}/train/${VID_ID}.json" ]]; then
        SPLIT_DIR="${FRAMES_DIR}/train"
    elif [[ -f "${ANN_DIR}/test/${VID_ID}.json" ]]; then
        SPLIT_DIR="${FRAMES_DIR}/test"
    else
        SPLIT_DIR="${FRAMES_DIR}"
    fi
    OUT="${SPLIT_DIR}/${VID_ID}"

    if [[ -d "${OUT}" && -n "$(ls -A "${OUT}" 2>/dev/null)" ]]; then
        echo "  [skip] ${VID_ID}"
        return 0
    fi

    STAGE_OUT="${STAGE_DIR}/${VID_ID}"
    mkdir -p "${STAGE_OUT}" "${SPLIT_DIR}"
    SKIP=$((30 / FPS))
    (( SKIP < 1 )) && SKIP=1   # FPS > 30: keep every source frame

    if ffmpeg -i "${VID}" -vf "select='not(mod(n,${SKIP}))'" \
              -vsync vfr -frame_pts true -q:v 2 \
              "${STAGE_OUT}/%06d.jpg" -loglevel error 2>/dev/null; then
        # Move from local fast NVMe to final FRAMES_DIR (one mv per video,
        # not per jpg → minimises Lustre metadata ops).
        mv "${STAGE_OUT}" "${OUT}"
        echo "  [done] ${VID_ID}"
    else
        echo "  [fail] ${VID_ID}" >&2
        echo "${VID_ID}" >> "${FAILED_LOG}"
        rm -rf "${STAGE_OUT}" "${OUT}"
    fi
}
export -f extract_one
export FPS ANN_DIR FRAMES_DIR STAGE_DIR FAILED_LOG

find "${TMPVID}" -name "*.mp4" -print0 \
    | xargs -0 -n1 -P "${JOBS}" -I{} bash -c 'extract_one "$@"' _ {}

VIDEO_COUNT=$(find "${FRAMES_DIR}" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l)
FAIL_COUNT=$(wc -l < "${FAILED_LOG}" 2>/dev/null || echo 0)

# Only delete tmp if fully extracted; keep for retry otherwise
if [[ ${VIDEO_COUNT} -eq 1000 ]]; then
    rm -rf "${TMPVID}"
elif [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo "  [warn] ${FAIL_COUNT} videos failed. See ${FAILED_LOG}. Re-run to retry, or use --force to re-download zips."
fi

echo ""
echo "Done. Annotations: ${ANN_COUNT:-1000} | Videos Extracted: ${VIDEO_COUNT}/1000"
