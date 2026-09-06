#!/usr/bin/env bash
# Build the oracle exports M1/M2/M3 are calibrated against.
#
#   bash experiments/M_evaluation_methods/build_oracle_dataset.sh
#   N_CLIPS=40 bash experiments/M_evaluation_methods/build_oracle_dataset.sh
#
# Any dataset whose annotations read through tools/planner/oracle.py works, not
# only VidVRD. VidOR uses the identical format, so:
#
#   CLIPS_FILE=eval/vidor_winnable_w16.txt \
#   ANN_DIR=data/video/vidor/annotations/training \
#   OUT_DIR=eval/probe/vidor N_CLIPS=25 \
#     bash experiments/M_evaluation_methods/build_oracle_dataset.sh
#
# These are the CEILING: latents built straight from ground-truth boxes, with
# no model. Whatever a trained model scores, it scores against these.
#
# Reading VidVRD annotations needs pillow, which the training venv (python 3.6,
# pinned by tensorflow 1.15) does not carry. `.venv-local` does.

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

N_CLIPS="${N_CLIPS:-20}"
CLIPS_FILE="${CLIPS_FILE:-eval/vidvrd_winnable_clips.txt}"
ANN_DIR="${ANN_DIR:-data/video/vidvrd/annotations/train}"
OUT_DIR="${OUT_DIR:-eval/probe/batch}"
MEM_KB="${MEM_KB:-6000000}"

PY="python3"
for cand in .venv-local/bin/python3 venv/bin/python3; do
    if [[ -x "${cand}" ]] && "${cand}" -c 'import PIL' 2>/dev/null; then
        PY="${cand}"; break
    fi
done
if ! "${PY}" -c 'import PIL' 2>/dev/null; then
    echo "FATAL: no python with pillow. Try: .venv-local/bin/pip install pillow" >&2
    exit 3
fi
echo "using ${PY}"

if [[ ! -f "${CLIPS_FILE}" ]]; then
    echo "FATAL: ${CLIPS_FILE} missing. Produce it with:" >&2
    echo "  python3 tools/video/screen_vidvrd.py --winnable-only --no-fill-only \\" >&2
    echo "      --min-frames 45 --list ${CLIPS_FILE}" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"
built=0; skipped=0; failed=0

while read -r clip; do
    clip="$(echo "${clip}" | tr -d '\r')"
    [[ -z "${clip}" ]] && continue
    (( built + skipped >= N_CLIPS )) && break

    # A clip id may carry a subdirectory, as VidOR's do ("0021/6833795682").
    # The output name flattens it so every export lands in one directory.
    ANN="${ANN_DIR}/${clip}.json"
    out_name="${clip//\//-}"
    if [[ ! -f "${ANN}" ]]; then
        echo "  no annotation: ${clip}"; continue
    fi
    if [[ -f "${OUT_DIR}/${out_name}.npz" ]]; then
        skipped=$((skipped + 1)); continue
    fi

    # --no-fill is deliberate: filling fabricates transitions, and M3 measures
    # that a filled clip is one it cannot say anything about.
    # stdout is suppressed and stderr is NOT. The line below used to read
    # `>/dev/null 2>&1`, so a clip that failed printed "export failed" and
    # nothing about why -- including when the cause was the memory cap, which
    # looks identical to a malformed annotation.
    if ( ulimit -v "${MEM_KB}"
         "${PY}" tools/planner/oracle.py "${ANN}" \
             --out "${OUT_DIR}/${out_name}.npz" --max-objects 3 --no-fill \
             >/dev/null ); then
        built=$((built + 1))
    else
        echo "  export failed: ${clip}"; failed=$((failed + 1))
    fi
done < "${CLIPS_FILE}"

total="$(ls -1 "${OUT_DIR}"/*.npz 2>/dev/null | wc -l)"

echo
echo "built ${built}, already had ${skipped}, failed ${failed}"
echo "total exports: ${total}"

# An empty output directory is a failure, and it used to exit 0. The callers
# read that as "the dataset is built": experiments/M4_temporal_distance and
# sh/dataset.sh both run this and carry straight on, so a run where every
# clip failed reached the scorer as an empty dataset rather than as an error.
if (( total == 0 )); then
    echo >&2
    echo "FATAL: ${OUT_DIR} holds no export, so nothing was built." >&2
    echo "       ${failed} clip(s) failed; their stderr is above." >&2
    exit 4
fi

echo
echo "next:  bash experiments/M_evaluation_methods/run_local.sh"
