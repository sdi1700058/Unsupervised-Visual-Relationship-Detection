#!/usr/bin/env bash
# Score planner exports on the frame interpolation task.
#
# Slides a window of K frames across each export. For every window the planner
# gets the first and last frame and has to reconstruct the K-2 frames between
# them. One CSV row per (export, method, window).
#
# Needs numpy alone. Make the exports first, where the model lives:
#
#     python3 tools/planner/export_latents.py <model_dir> -o dog.npz
#
# Usage
#   bash tools/planner/eval_plannability.sh <export.npz | export_dir> [options]
#
#   --methods LIST    comma separated. Default: bfs,pddl,ama3.
#                     A method that is not installed is skipped, not fatal.
#   --window K        frames per window, K >= 3. Default: 5.
#   --stride S        gap between window starts. Default: K-1 (no overlap).
#   --max-windows N   cap per export. Useful for a quick look. Default: all.
#   --budget SECONDS  planner time limit per window. Default: 60.
#   --length-mode M   max (shortest plan within k-1 actions, the default),
#                     exact (exactly k-1), or free (unbounded). Trained models
#                     merge frames into one latent, so exact is usually
#                     unsatisfiable for them; the oracle export wants exact.
#   --name NAME       output directory name. Default: taken from the input.
#
# Output
#   eval/planner/<name>/summary.csv

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 ]]; then
    sed -n '2,26p' "$0" | sed 's/^# \?//'
    exit 1
fi

INPUT="$1"; shift
METHODS="bfs,pddl,ama3"
WINDOW=5
STRIDE=""
MAX_WINDOWS=""
BUDGET=60
LENGTH_MODE="max"
NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)     METHODS="$2"; shift 2 ;;
        --window)      WINDOW="$2"; shift 2 ;;
        --stride)      STRIDE="$2"; shift 2 ;;
        --max-windows) MAX_WINDOWS="$2"; shift 2 ;;
        --budget)      BUDGET="$2"; shift 2 ;;
        --length-mode) LENGTH_MODE="$2"; shift 2 ;;
        --name)        NAME="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -z "${STRIDE}" ]] && STRIDE=$(( WINDOW - 1 ))

# One export or a directory of them.
EXPORTS=()
if [[ -d "${INPUT}" ]]; then
    while IFS= read -r f; do EXPORTS+=("$f"); done \
        < <(find "${INPUT}" -maxdepth 1 -name '*.npz' | sort)
    [[ -z "${NAME}" ]] && NAME="$(basename "${INPUT}")"
elif [[ -f "${INPUT}" ]]; then
    EXPORTS+=("$(cd "$(dirname "${INPUT}")" && pwd)/$(basename "${INPUT}")")
    [[ -z "${NAME}" ]] && NAME="$(basename "${INPUT}" .npz)"
else
    echo "no such export or directory: ${INPUT}" >&2
    exit 1
fi

if [[ ${#EXPORTS[@]} -eq 0 ]]; then
    echo "no .npz exports found in ${INPUT}" >&2
    exit 1
fi

SUMMARY_DIR="${PROJECT_DIR}/eval/planner/${NAME}"
SUMMARY_CSV="${SUMMARY_DIR}/summary.csv"
LOG_DIR="${SUMMARY_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "exports  ${#EXPORTS[@]}"
echo "methods  ${METHODS}"
echo "window   K=${WINDOW} stride=${STRIDE}"
echo "length   ${LENGTH_MODE}"
echo "summary  ${SUMMARY_CSV}"
echo

echo "export,method,init,goal,reachability,plan_length,expected_length,moving_steps,moving_gt_steps,skipped_absent,length_match,bbox_mse,baseline_mse,mse_ratio,beats_baseline,bbox_iou,baseline_iou,temporal_order,decode_fallbacks,wall_s" \
    > "${SUMMARY_CSV}"

IFS=',' read -ra METHOD_LIST <<< "${METHODS}"

for EXPORT in "${EXPORTS[@]}"; do
    STEM="$(basename "${EXPORT}" .npz)"

    WINDOWS="$(python3 - "${EXPORT}" "${WINDOW}" "${STRIDE}" "${MAX_WINDOWS}" <<'PY'
import sys
sys.path.insert(0, ".")
from tools.planner.common.export import load
from tools.planner.common.windows import make_windows

export, k, stride, cap = sys.argv[1:5]
n_frames = len(load(export))
for w in make_windows(n_frames, int(k), int(stride), int(cap) if cap else None):
    print(w["init"], w["goal"])
PY
)" || { echo "  cannot build windows for ${STEM}, skipping" >&2; continue; }

    while read -r INIT GOAL; do
        [[ -z "${INIT}" ]] && continue

        for METHOD in "${METHOD_LIST[@]}"; do
            OUT_DIR="${SUMMARY_DIR}/${STEM}/${METHOD}/win_${INIT}_${GOAL}"
            LOG="${LOG_DIR}/${STEM}_${METHOD}_${INIT}_${GOAL}.log"

            set +e
            python3 tools/planner/plan_video.py "${EXPORT}" \
                --method "${METHOD}" \
                --init "${INIT}" --goal "${GOAL}" \
                --time-budget-s "${BUDGET}" \
                --length-mode "${LENGTH_MODE}" \
                --out-dir "${OUT_DIR}" \
                > "${LOG}" 2>&1
            RC=$?
            set -e

            METRICS="${OUT_DIR}/metrics.json"
            if [[ ${RC} -ne 0 || ! -f "${METRICS}" ]]; then
                echo "  ${STEM} ${METHOD} ${INIT}->${GOAL}  skipped (rc=${RC})"
                echo "${STEM},${METHOD},${INIT},${GOAL},false,0,,,,,,,,,,,,,," >> "${SUMMARY_CSV}"
                continue
            fi

            python3 - "${METRICS}" "${STEM}" "${METHOD}" <<'PY' >> "${SUMMARY_CSV}"
import json, sys

path, stem, method = sys.argv[1:4]
m = json.load(open(path))
cell = lambda k: "" if m.get(k) is None else m[k]

row = [stem, method, cell("init_frame"), cell("goal_frame"),
       cell("reachability"), cell("plan_length"), cell("expected_plan_length"),
       cell("moving_steps"), cell("moving_gt_steps"), cell("skipped_absent"), cell("plan_length_match"), cell("bbox_mse_mean"),
       cell("baseline_mse_mean"), cell("mse_ratio"), cell("beats_baseline"),
       cell("bbox_iou_mean"), cell("baseline_iou_mean"),
       cell("temporal_order"), cell("decode_fallbacks"), cell("wall_s")]
print(",".join(str(x) for x in row))

ratio = m.get("mse_ratio")
verdict = "" if ratio is None else f" ratio {ratio:.3f}"
print(f"  {stem} {method} {m.get('init_frame')}->{m.get('goal_frame')}"
      f"  plan {m.get('plan_length')}{verdict}", file=sys.stderr)
PY
        done
    done <<< "${WINDOWS}"
done

ROWS=$(( $(wc -l < "${SUMMARY_CSV}") - 1 ))
echo
echo "wrote ${ROWS} rows to ${SUMMARY_CSV}"
echo "plot with: python3 tools/planner/viz_plannability.py ${SUMMARY_DIR}"
