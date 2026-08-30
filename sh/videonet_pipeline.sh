#!/usr/bin/env bash
# VideoNet end to end: clips in, scoreable exports out.
#
#   bash sh/videonet_pipeline.sh
#   VARIANT=B OBJECT_PROMPT=pen bash sh/videonet_pipeline.sh
#
# VideoNet has NO bounding boxes and NO relation annotations -- it is a video
# question-answering benchmark, 37 domains, 95.4 GB, YouTube-sourced, with
# yes/no and multiple-choice questions and an action label per clip. So the
# boxes have to be manufactured, and that is what this does.
#
# Why bother, given that cost: its 37 domains are DOMAIN-SPECIFIC ACTION, which
# scores well on Criterion 0. Skill video is rule-governed in the way
# blocksworld is and VidVRD is not, and Criterion 0 outranks annotation
# density.
#
# The chain, and every link is already written:
#
#   1. sh/download_videonet.sh   MP4 -> frames at N fps
#   2. sh/setup_venv_detect.sh   the py>=3.9 sidecar (MediaPipe needs it; the
#                                training venv is py3.6, pinned by tf 1.15)
#   3. tools/synth_bbox.py       frames -> VidVRD-schema JSON with tids
#   4. tools/planner/oracle.py   JSON -> planner export
#   5. tools/planner/plan_validity.py    score it
#
# Step 5 is M3, and M3 is the reason this is viable at all. It needs no ground
# truth and no relation labels, which is exactly what auto-annotated data
# lacks. M1 and M2 CANNOT run on VideoNet: they need human relation labels and
# VideoNet has none.
#
# The contract between steps 3 and 4 is covered by
# `tools/planner/tests/test_oracle.py::TestSynthBboxContract`, which runs
# without MediaPipe or any download, so this pipeline cannot rot in silence.
#
# WHAT YOU MUST SUPPLY, because none of it can be downloaded from here:
#
#   * the clips themselves, as data/video/videonet/videos/<stem>.mp4.
#     VideoNet's training parquets carry only a YouTube id and a timespan;
#     the videos/ folder on Hugging Face is benchmark-only.
#   * one run of sh/setup_venv_detect.sh, which installs MediaPipe and
#     transformers into venv-detect and pulls the Grounded-DINO checkpoint.

set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

FPS="${FPS:-5}"
VARIANT="${VARIANT:-A}"
OBJECT_PROMPT="${OBJECT_PROMPT:-}"
MEM_KB="${MEM_KB:-6000000}"

DATA="data/video/videonet"
FRAMES="${DATA}/frames_${FPS}fps"
ANN="${DATA}/annotations/${VARIANT}/train"
EXPORTS="eval/exports/videonet"

step () { echo; echo "=========================================="; echo "$*"; echo "=========================================="; }

# ── 1. frames ───────────────────────────────────────────────────────────────
step "1  frames at ${FPS} fps"
shopt -s nullglob
MP4S=("${DATA}/videos"/*.mp4)
shopt -u nullglob
if (( ${#MP4S[@]} == 0 )); then
    cat >&2 <<'MSG'
No MP4s found.

Put clips in data/video/videonet/videos/<stem>.mp4 first. VideoNet's parquets
give only a YouTube id and a timespan, so the clips have to be fetched by
hand; nothing here can do it for you.

Pick from a domain whose world has rules -- pen spinning, knot tying, card
shuffling. That is the whole reason for choosing VideoNet over VidVRD.
MSG
    exit 2
fi
FPS="${FPS}" bash sh/download_videonet.sh

# ── 2. sidecar venv ─────────────────────────────────────────────────────────
step "2  detection sidecar"
if [[ ! -x venv-detect/bin/python ]]; then
    echo "venv-detect missing. Run this once, then re-run this script:" >&2
    echo "    bash sh/setup_venv_detect.sh" >&2
    exit 3
fi
echo "have venv-detect"

# ── 3. synthesise boxes ─────────────────────────────────────────────────────
step "3  boxes from MediaPipe (+ Grounded-DINO if a prompt is given)"
mkdir -p "${ANN}"
n_ann=0
for d in "${FRAMES}"/*/; do
    [[ -d "${d}" ]] || continue
    stem="$(basename "${d}")"
    out="${ANN}/${stem}.json"
    if [[ -f "${out}" ]]; then echo "  have ${stem}"; n_ann=$((n_ann+1)); continue; fi
    echo "  annotate ${stem}"
    ARGS=(--frames-dir "${d}" --video-id "${stem}" --variant "${VARIANT}" --out "${out}")
    [[ -n "${OBJECT_PROMPT}" ]] && ARGS+=(--object-prompt "${OBJECT_PROMPT}")
    if venv-detect/bin/python tools/synth_bbox.py "${ARGS[@]}"; then
        n_ann=$((n_ann+1))
    else
        echo "    FAILED ${stem}" >&2
    fi
done
(( n_ann == 0 )) && { echo "no annotations produced" >&2; exit 4; }

# ── 4. exports ──────────────────────────────────────────────────────────────
step "4  exports"
mkdir -p "${EXPORTS}"
PY="python3"
for cand in .venv-local/bin/python3 venv/bin/python3; do
    [[ -x "${cand}" ]] && "${cand}" -c 'import PIL' 2>/dev/null && { PY="${cand}"; break; }
done
for j in "${ANN}"/*.json; do
    [[ -f "${j}" ]] || continue
    stem="$(basename "${j}" .json)"
    [[ -f "${EXPORTS}/${stem}.npz" ]] && continue
    # --no-fill: filling fabricates transitions, and M3 measures that a filled
    # clip is one it can say nothing about.
    ( ulimit -v "${MEM_KB}"
      "${PY}" tools/planner/oracle.py "${j}" --out "${EXPORTS}/${stem}.npz" \
          --max-objects 3 --no-fill ) || echo "  export failed: ${stem}" >&2
done

# ── 5. score, with the only metric that can ─────────────────────────────────
step "5  M3, the only method that works without annotation"
mkdir -p eval/validity/videonet
( ulimit -v "${MEM_KB}"
  python3 tools/planner/plan_validity.py "${EXPORTS}"/*.npz \
      --out-dir eval/validity/videonet )

cat <<'NOTE'

Read the separation column, not the validity column. A clip whose scrambled
trajectory scores as well as its real one is a clip the measure is SILENT on,
and its validity is not evidence. See experiments/M_evaluation_methods/README.md.

M1 and M2 cannot run here: VideoNet has no human relation labels.
NOTE
