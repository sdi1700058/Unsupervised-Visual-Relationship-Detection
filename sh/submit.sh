#!/usr/bin/env bash
# submit.sh — Convenience wrapper to launch a SLURM training job on Sherlock.
#
# All experiment knobs are env-var overridable. Defaults run an all-categories
# vidvrd training job at fps=3 with U=40 A=2 P=20.
#
# Examples
# --------
#   bash sh/submit.sh                                   # default vidvrd run
#   CATEGORY=person bash sh/submit.sh                   # per-category
#   FPS=5 EPOCH=8000 bash sh/submit.sh                  # 5fps, longer training
#   DOMAIN=labeled_objects ARGS_TAIL="None sequential 5000 5000 9000" \
#       bash sh/submit.sh                               # different domain
#   GPUS=2 MEM=64G TIME=48:00:00 bash sh/submit.sh      # bigger SLURM ask
#
# Custom raw command (bypasses the vidvrd template):
#   TRAIN_CMD="python3 strips.py learn puzzle mnist 3 3 100" bash sh/submit.sh
#
# All paths and outputs stay inside the project directory.

set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Training-side params (used to build TRAIN_CMD if not supplied) ───────────
DOMAIN="${DOMAIN:-vidvrd}"
AECLASS="${AECLASS:-FirstOrderSAE}"
U="${U:-40}"
A="${A:-2}"
P="${P:-20}"
EPOCH="${EPOCH:-5000}"
MAX_VIDEOS="${MAX_VIDEOS:-None}"
BATCH="${BATCH:-None}"
FPS="${FPS:-30}"
CATEGORY="${CATEGORY:-bicycle}"
TRANSITION_MODE="${TRANSITION_MODE:-sequential}"

# Build the default vidvrd training command.
# Positional: aeclass U A P ann_dir frames_dir transition_mode epoch max_videos batch_size fps category
DEFAULT_VIDVRD_CMD="python3 strips.py learn ${DOMAIN} ${AECLASS} ${U} ${A} ${P} None None ${TRANSITION_MODE} ${EPOCH} ${MAX_VIDEOS} ${BATCH} ${FPS} ${CATEGORY}"

# Allow callers to provide TRAIN_CMD directly (overrides the template) or
# ARGS_TAIL to append/replace tail of the strips.py call (advanced).
if [[ -n "${ARGS_TAIL:-}" ]]; then
    TRAIN_CMD="${TRAIN_CMD:-python3 strips.py learn ${DOMAIN} ${AECLASS} ${U} ${A} ${P} ${ARGS_TAIL}}"
else
    TRAIN_CMD="${TRAIN_CMD:-${DEFAULT_VIDVRD_CMD}}"
fi

# ── SLURM-side params ────────────────────────────────────────────────────────
JOB_TAG="${DOMAIN}"
[[ "${CATEGORY}" != "None" && -n "${CATEGORY}" ]] && JOB_TAG="${DOMAIN}-${CATEGORY}"
# Unique per submission so concurrent jobs never collide in squeue / logs.
# Override with JOB_NAME=... or set JOB_SUFFIX="" to disable the suffix.
JOB_SUFFIX="${JOB_SUFFIX-$(date +%Y%m%d-%H%M%S)-$$}"
JOB_NAME="${JOB_NAME:-fosae-${JOB_TAG}${JOB_SUFFIX:+-${JOB_SUFFIX}}}"
PARTITION="${PARTITION:-gpu}"

# Auto-estimate MEM/TIME/CONSTRAINT from dataset size (vidvrd only).
# Set AUTO_RESOURCES=0 to disable. Any explicitly-set var is preserved.
if [[ "${AUTO_RESOURCES:-1}" == "1" && "${DOMAIN}" == "vidvrd" ]]; then
    if _EST="$(DOMAIN="${DOMAIN}" CATEGORY="${CATEGORY}" FPS="${FPS}" \
                EPOCH="${EPOCH}" \
                BATCH="${BATCH/None/1000}" \
                FORMAT=env bash "${PROJECT_DIR}/sh/estimate_resources.sh" 2>/dev/null)"; then
        # Apply only when caller didn't override.
        eval "_${_EST}"
        : "${MEM:=${_MEM:-}}"
        : "${TIME:=${_TIME:-}}"
        : "${GPUS:=${_GPUS:-}}"
        : "${CPUS:=${_CPUS:-}}"
        : "${CONSTRAINT:=${_CONSTRAINT:-}}"
        echo "[submit] auto-estimate: states=${_EST_NUM_STATES:-?} trans=${_EST_NUM_TRANS:-?}"
    fi
fi

GPUS="${GPUS:-1}"
CPUS="${CPUS:-2}"
MEM="${MEM:-32G}"
TIME="${TIME:-24:00:00}"
CONSTRAINT="${CONSTRAINT:-}"
MAIL_TYPE="${MAIL_TYPE:-}"
MAIL_USER="${MAIL_USER:-}"

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/out"

# Auto-promote QoS to "long" when TIME implies > 2 days (Sherlock default cap).
QOS="${QOS:-}"
if [[ -z "${QOS}" ]]; then
    case "${TIME}" in
        [3-9]-*|[1-9][0-9]-*) QOS="long" ;;   # >= 3-00:00:00
    esac
fi

# Pass training-side params through to run_training.sh so post-train hooks
# (extract_fol + visualize_fol) can resolve OUT_DIR.
EXPORT_VARS="ALL"
EXPORT_VARS+=",TRAIN_CMD=${TRAIN_CMD}"
EXPORT_VARS+=",DOMAIN=${DOMAIN},AECLASS=${AECLASS}"
EXPORT_VARS+=",U=${U},A=${A},P=${P}"
EXPORT_VARS+=",CATEGORY=${CATEGORY}"
EXPORT_VARS+=",EXTRACT_FOL=${EXTRACT_FOL:-1},VISUALIZE=${VISUALIZE:-1},VIS_NUM=${VIS_NUM:-6}"

SBATCH_ARGS=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --gpus="${GPUS}"
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --time="${TIME}"
    --output="${PROJECT_DIR}/logs/%x.%j.out"
    --error="${PROJECT_DIR}/logs/%x.%j.err"
    --export="${EXPORT_VARS}"
)

[[ -n "${QOS}" ]]        && SBATCH_ARGS+=(--qos="${QOS}")
[[ -n "${CONSTRAINT}" ]] && SBATCH_ARGS+=(--constraint="${CONSTRAINT}")
[[ -n "${MAIL_TYPE}" ]]  && SBATCH_ARGS+=(--mail-type="${MAIL_TYPE}")
[[ -n "${MAIL_USER}" ]]  && SBATCH_ARGS+=(--mail-user="${MAIL_USER}")

echo "[submit] Job name : ${JOB_NAME}"
echo "[submit] Partition: ${PARTITION}  GPUs=${GPUS}  Mem=${MEM}  Time=${TIME}  QoS=${QOS:-default}"
echo "[submit] Cmd      : ${TRAIN_CMD}"
echo "[submit] Hooks    : EXTRACT_FOL=${EXTRACT_FOL:-1}  VISUALIZE=${VISUALIZE:-1}"
echo "[submit] Logs     : ${PROJECT_DIR}/logs/${JOB_NAME}.<JOBID>.{out,err}"

# Dry-run mode: print sbatch invocation without submitting.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[submit] DRY_RUN=1 — would run:"
    echo "  sbatch ${SBATCH_ARGS[*]} ${PROJECT_DIR}/run_training.sh"
    exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "${PROJECT_DIR}/run_training.sh"
