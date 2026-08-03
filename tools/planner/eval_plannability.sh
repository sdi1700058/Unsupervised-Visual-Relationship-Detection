#!/usr/bin/env bash
# tools/planner/eval_plannability.sh — SPEC §T H3 batch driver.
#
# Iterates over videos × available planner routes for a trained model_dir,
# invokes `tools/planner/plan_video.py --route {a,b,c}` per (video, route),
# and aggregates the metrics.json files into a single summary.csv.
#
# Usage
#   bash tools/planner/eval_plannability.sh <model_dir> [videos_dir] [routes] [pairs]
#
#   model_dir   : path to trained FirstOrderSAE (contains net0.h5 + loaded_videos.json).
#   videos_dir  : dir with baked overfit npz files. Default: dir of manifest.npz_path.
#   routes      : comma-separated list of routes to run. Default: `a,b,c`.
#                 The script auto-skips a route whose entry raises NotImplementedError.
#   pairs       : comma-separated `start:goal` pairs. Default: `0:-1`.
#
# Output
#   eval/planner/<model_stem>/summary.csv
#     columns: video_id,route,start,goal,reachability,plan_length,wall_s,bbox_mse_mean,n_deltas
#
# SPEC.md §C C17 (all-route metrics parity), §V V15 (determinism per route).

set -euo pipefail

MODEL_DIR="${1:-}"
if [[ -z "${MODEL_DIR}" ]]; then
    echo "usage: $0 <model_dir> [videos_dir] [routes] [pairs]" >&2
    exit 1
fi
MODEL_DIR="$(cd "${MODEL_DIR}" && pwd)"

VIDEOS_DIR="${2:-}"
ROUTES="${3:-a,b,c}"
PAIRS="${4:-0:-1}"
TIME_BUDGET_S="${TIME_BUDGET_S:-60}"

MODEL_STEM="$(basename "${MODEL_DIR}")"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUMMARY_DIR="${PROJECT_ROOT}/eval/planner/${MODEL_STEM}"
SUMMARY_CSV="${SUMMARY_DIR}/summary.csv"

mkdir -p "${SUMMARY_DIR}"

# --- 1. Collect video npz paths ---
if [[ -z "${VIDEOS_DIR}" ]]; then
    if [[ ! -f "${MODEL_DIR}/loaded_videos.json" ]]; then
        echo "ERROR: no loaded_videos.json at ${MODEL_DIR} and no videos_dir arg. Pass one explicitly." >&2
        exit 2
    fi
    _NPZ="$(python3 -c "import json; m=json.load(open('${MODEL_DIR}/loaded_videos.json')); print(m.get('npz_path',''))")"
    if [[ -z "${_NPZ}" ]]; then
        echo "ERROR: loaded_videos.json missing 'npz_path'" >&2
        exit 2
    fi
    VIDEOS_DIR="$(dirname "${_NPZ}")"
fi

shopt -s nullglob
NPZS=("${VIDEOS_DIR}"/*.npz)
shopt -u nullglob
if (( ${#NPZS[@]} == 0 )); then
    echo "ERROR: no *.npz in ${VIDEOS_DIR}" >&2
    exit 3
fi
echo "[eval_plannability] found ${#NPZS[@]} video npz files in ${VIDEOS_DIR}"

# --- 2. Header ---
echo "video_id,route,start,goal,reachability,plan_length,wall_s,bbox_mse_mean,n_deltas" > "${SUMMARY_CSV}"

# --- 3. Iterate ---
IFS=',' read -ra ROUTE_LIST <<< "${ROUTES}"
IFS=',' read -ra PAIR_LIST <<< "${PAIRS}"

for NPZ in "${NPZS[@]}"; do
    VID_STEM="$(basename "${NPZ}" .npz)"
    for PAIR in "${PAIR_LIST[@]}"; do
        START="${PAIR%%:*}"
        GOAL="${PAIR##*:}"
        for ROUTE in "${ROUTE_LIST[@]}"; do
            echo "[eval_plannability] ${VID_STEM} route=${ROUTE} start=${START} goal=${GOAL}"
            OUT_DIR="${SUMMARY_DIR}/${VID_STEM}/route_${ROUTE}/plan_${START}_${GOAL}"
            mkdir -p "${OUT_DIR}"
            set +e
            python3 "${PROJECT_ROOT}/tools/planner/plan_video.py" "${MODEL_DIR}" \
                --route "${ROUTE}" \
                --npz-path "${NPZ}" \
                --start "${START}" --goal "${GOAL}" \
                --time-budget-s "${TIME_BUDGET_S}" \
                > "${OUT_DIR}/stdout.log" 2> "${OUT_DIR}/stderr.log"
            RC=$?
            set -e

            METRICS_JSON="${OUT_DIR}/metrics.json"
            if [[ ${RC} -eq 2 || ! -f "${METRICS_JSON}" ]]; then
                echo "  route ${ROUTE}: SKIPPED or FAILED (rc=${RC})"
                echo "${VID_STEM},${ROUTE},${START},${GOAL},false,0,0,," >> "${SUMMARY_CSV}"
                continue
            fi

            # Extract metrics into one CSV row via python.
            python3 - "${METRICS_JSON}" "${VID_STEM}" "${ROUTE}" "${START}" "${GOAL}" <<'PY' >> "${SUMMARY_CSV}"
import json, sys
mpath, vid, route, start, goal = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
with open(mpath) as f:
    m = json.load(f)
reach = str(m.get("reachability", False)).lower()
plen = m.get("plan_length", 0)
wall = m.get("wall_s", 0.0)
bmse = m.get("bbox_mse_mean", "")
ndel = m.get("n_deltas", m.get("n_unique_deltas", ""))
print(f"{vid},{route},{start},{goal},{reach},{plen},{wall:.3f},{bmse},{ndel}")
PY
        done
    done
done

echo "[eval_plannability] wrote ${SUMMARY_CSV}"
wc -l "${SUMMARY_CSV}"
