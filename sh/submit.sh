#!/usr/bin/env bash
# submit.sh — Launch a SLURM training job on Sherlock.
#
# Composes the canonical hierarchical output dir
#   out/<domain>/<category>/<run_tag>/        (SPEC §C15, §V13)
# via the shared python helper (latplan.util.paths.resolved_out_dir) so
# strips.py + run_training.sh + this script all see the SAME path.
#
# Env knobs (defaults shown):
#   DOMAIN=vidvrd            puzzle | blocks | vidvrd | actiongenome | labeled_objects
#   AECLASS=FirstOrderSAE    model class
#   U=40 A=2 P=20            FOSAE hyperparams
#   EPOCH=5000               training epochs
#   BATCH=None               batch size override (None = strips.py default)
#   FPS=3                    frame-rate (video domains only)
#   CATEGORY=bicycle         video class filter (None = all)
#   TRANSITION_MODE=sequential
#   MAX_VIDEOS=None
#   PUZZLE_TYPE=mnist WIDTH=3 HEIGHT=3   (puzzle only)
#   TRACK=blocks-5-3                      (blocks only)
#
# Examples
#   DOMAIN=puzzle PUZZLE_TYPE=mnist WIDTH=3 HEIGHT=3 EPOCH=1000 bash sh/submit.sh
#   DOMAIN=blocks TRACK=blocks-5-3 EPOCH=1000 MEM=64G bash sh/submit.sh
#   DOMAIN=vidvrd CATEGORY=bicycle FPS=30 bash sh/submit.sh
#   DOMAIN=actiongenome CATEGORY=chair bash sh/submit.sh
#   DOMAIN=labeled_objects MAX_IMAGES=5000 bash sh/submit.sh
#
# Override TRAIN_CMD to bypass the per-domain template entirely:
#   TRAIN_CMD="python3 strips.py learn puzzle FirstOrderAE mnist 3 3" bash sh/submit.sh

set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Training-side params ─────────────────────────────────────────────────────
DOMAIN="${DOMAIN:-vidvrd}"
AECLASS="${AECLASS:-FirstOrderSAE}"
U="${U:-40}"
A="${A:-2}"
P="${P:-20}"
EPOCH="${EPOCH:-5000}"
BATCH="${BATCH:-None}"
TRANSITION_MODE="${TRANSITION_MODE:-sequential}"
MAX_VIDEOS="${MAX_VIDEOS:-None}"
MAX_IMAGES="${MAX_IMAGES:-None}"

# Video-only knobs
# When NPZ_PATH points at a baked overfit npz, read fps from its meta so the
# OUT_DIR tag reflects the actual data rather than the env-var default.
if [[ -n "${NPZ_PATH:-}" && -f "${NPZ_PATH}" && -z "${FPS:-}" ]]; then
    FPS="$(python3 -c "
import numpy as np, json, sys
d = np.load('${NPZ_PATH}', allow_pickle=True)
m = d['meta'].item()
m = json.loads(m.decode('utf-8') if isinstance(m, bytes) else m)
print(m.get('fps', 3))
" 2>/dev/null || echo 3)"
fi
FPS="${FPS:-3}"
CATEGORY="${CATEGORY:-bicycle}"
# NPZ_PATH (video only): pre-baked overfit npz produced by `setup-dataset.py
# video_ag|video_vidvrd`. When set, strips.py skips on-the-fly build_dataset()
# and consumes the npz directly. Default `None` keeps legacy behaviour.
NPZ_PATH="${NPZ_PATH:-None}"

# Puzzle-only knobs
PUZZLE_TYPE="${PUZZLE_TYPE:-mnist}"
WIDTH="${WIDTH:-3}"
HEIGHT="${HEIGHT:-3}"
NUM_EXAMPLES="${NUM_EXAMPLES:-20000}"

# Blocks-only knobs
TRACK="${TRACK:-blocks-5-3}"
NUM_TRANSITIONS="${NUM_TRANSITIONS:-6500}"

# ── Per-domain TRAIN_CMD template (only used if TRAIN_CMD unset) ─────────────
# Positional signatures (from strips.py):
#   puzzle(aeclass, type, width, height, U, A, P, num_examples)
#   blocksworld(aeclass, track, U, A, P, num_examples)
#   labeled_objects(aeclass, U, A, P, num_objects, comment, dataset_path,
#                   images_dir, transition_mode, epoch, max_images, batch_size)
#   vidvrd(aeclass, U, A, P, annotations_dir, frames_dir, transition_mode,
#          epoch, max_videos, batch_size, fps, category)
#   actiongenome(aeclass, U, A, P, annotations_dir, frames_dir, transition_mode,
#                epoch, max_videos, batch_size, fps, category)
build_default_cmd() {
    case "${DOMAIN}" in
        puzzle)
            echo "python3 strips.py learn puzzle ${AECLASS} ${PUZZLE_TYPE} ${WIDTH} ${HEIGHT} ${U} ${A} ${P} ${NUM_EXAMPLES}"
            ;;
        blocks|blocksworld)
            echo "python3 strips.py learn blocksworld ${AECLASS} ${TRACK} ${U} ${A} ${P} ${NUM_TRANSITIONS}"
            ;;
        labeled_objects)
            echo "python3 strips.py learn labeled_objects ${AECLASS} ${U} ${A} ${P} None None None None ${TRANSITION_MODE} ${EPOCH} ${MAX_IMAGES} ${BATCH}"
            ;;
        vidvrd)
            # NPZ_PATH overrides category — pass None so strips.py reads it from npz meta
            local _cat="${CATEGORY}"
            [[ -n "${NPZ_PATH}" && "${NPZ_PATH}" != "None" ]] && _cat="None"
            echo "python3 strips.py learn vidvrd ${AECLASS} ${U} ${A} ${P} None None ${TRANSITION_MODE} ${EPOCH} ${MAX_VIDEOS} ${BATCH} ${FPS} ${_cat} ${NPZ_PATH}"
            ;;
        actiongenome)
            local _cat="${CATEGORY}"
            [[ -n "${NPZ_PATH}" && "${NPZ_PATH}" != "None" ]] && _cat="None"
            echo "python3 strips.py learn actiongenome ${AECLASS} ${U} ${A} ${P} None None ${TRANSITION_MODE} ${EPOCH} ${MAX_VIDEOS} ${BATCH} ${FPS} ${_cat} ${NPZ_PATH}"
            ;;
        *)
            echo "[submit] unknown DOMAIN=${DOMAIN}" >&2
            exit 2
            ;;
    esac
}

if [[ -n "${ARGS_TAIL:-}" ]]; then
    TRAIN_CMD="${TRAIN_CMD:-python3 strips.py learn ${DOMAIN} ${AECLASS} ${U} ${A} ${P} ${ARGS_TAIL}}"
else
    TRAIN_CMD="${TRAIN_CMD:-$(build_default_cmd)}"
fi

# ── Compose canonical JOB_OUT_DIR via shared helper (SPEC §C15, §V13) ────────
# Pass FULL parameters dict so the sha1 suffix differentiates lr/preencoder/etc.
JOB_OUT_DIR="$(
PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}" python3 - <<PY
import os, importlib.util
spec = importlib.util.spec_from_file_location("paths", "${PROJECT_DIR}/latplan/util/paths.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
params = {
    "U":[${U}], "A":[${A}], "P":[${P}],
    "epoch":[${EPOCH}], "batch":["${BATCH}"],
    "transition_mode":["${TRANSITION_MODE}"],
    "max_videos":["${MAX_VIDEOS}"], "max_images":["${MAX_IMAGES}"],
    "fps":[${FPS}], "category":["${CATEGORY}"],
    "puzzle_type":["${PUZZLE_TYPE}"], "width":[${WIDTH}], "height":[${HEIGHT}],
    "track":["${TRACK}"], "aeclass":["${AECLASS}"],
    "npz_path":["${NPZ_PATH}"],
    # Env-overridable hyperparameters — fold INTO the hash so jobs that differ
    # only on these knobs land in distinct OUT_DIRs (previously all collided
    # into the same dir and overwrote each other's training_history.csv).
    "lr":["${LR:-}"],
    "zerosuppress":["${ZEROSUPPRESS:-}"],
    "zerosuppress_delay":["${ZEROSUPPRESS_DELAY:-}"],
    "max_temperature":["${MAX_TEMPERATURE:-}"],
    "min_temperature":["${MIN_TEMPERATURE:-}"],
    "dropout":["${DROPOUT:-}"],
    "noise":["${NOISE:-}"],
    "layer":["${LAYER:-}"],
    "beta":["${BETA:-}"],
    "preenc_layers":["${PREENC_LAYERS:-}"],
    "preenc_dim":["${PREENC_DIM:-}"],
    "no_earlystop":["${NO_EARLYSTOP:-}"],
}
dom_kw = dict(type="${PUZZLE_TYPE}", width=${WIDTH}, height=${HEIGHT}, track="${TRACK}")
# When loading a pre-baked overfit npz, anchor the run_tag to the npz stem
# (the meaningful identifier of which slice is being trained on) instead of
# the default-CATEGORY env. Hash still includes npz_path for full uniqueness.
_npz = "${NPZ_PATH}"
if _npz in ("", "None", "none"):
    vc  = "${CATEGORY}"
    vc  = vc if vc not in ("", "None", "none") else "_all"
else:
    import os.path as _osp
    vc  = _osp.splitext(_osp.basename(_npz))[0]
fps = ${FPS}
print(m.resolved_out_dir("${DOMAIN}", params, "${AECLASS}",
                         ${U}, ${A}, ${P},
                         video_category=vc, fps=fps,
                         base=os.environ.get("OUT_DIR") or os.path.join("${PROJECT_DIR}", "out"),
                         **dom_kw))
PY
)"

if [[ -z "${JOB_OUT_DIR}" ]]; then
    echo "[submit] FATAL: could not compose JOB_OUT_DIR" >&2
    exit 3
fi

# Make sure strips.py / grid_search.log (V13) point at the per-job dir too.
export JOB_OUT_DIR
export OUT_DIR="${JOB_OUT_DIR}"

# ── SLURM-side params ────────────────────────────────────────────────────────
# JOB_TAG = <domain>-<category> per SPEC §C15. Suffix with timestamp+PID so
# squeue / log filenames stay unique across concurrent launches.
JOB_TAG="${DOMAIN}"
if [[ "${NPZ_PATH}" != "None" && -n "${NPZ_PATH}" && "${DOMAIN}" =~ ^(vidvrd|actiongenome)$ ]]; then
    _NPZ_STEM="$(basename "${NPZ_PATH}" .npz)"
    JOB_TAG="${DOMAIN}-${_NPZ_STEM}"
elif [[ "${CATEGORY}" != "None" && -n "${CATEGORY}" && "${DOMAIN}" =~ ^(vidvrd|actiongenome)$ ]]; then
    JOB_TAG="${DOMAIN}-${CATEGORY}"
fi
JOB_SUFFIX="${JOB_SUFFIX-$(date +%Y%m%d-%H%M%S)-$$}"
JOB_NAME="${JOB_NAME:-fosae-${JOB_TAG}${JOB_SUFFIX:+-${JOB_SUFFIX}}}"
PARTITION="${PARTITION:-gpu}"

# Auto-estimate resources for vidvrd (sacct-driven). Other domains: caller sets.
if [[ "${AUTO_RESOURCES:-1}" == "1" && "${DOMAIN}" == "vidvrd" ]]; then
    # JOB_NAME gates Mode 1 of the estimator, which sizes a job from what
    # past runs of the same name actually used. It was set above but never
    # passed, so that mode has never once fired and every estimate has come
    # from the heuristic fallback.
    if _EST="$(DOMAIN="${DOMAIN}" CATEGORY="${CATEGORY}" FPS="${FPS}" \
                EPOCH="${EPOCH}" JOB_NAME="${JOB_NAME}" \
                BATCH="${BATCH/None/1000}" \
                FORMAT=env bash "${PROJECT_DIR}/sh/estimate_resources.sh" 2>/dev/null)"; then
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

QOS="${QOS:-}"
if [[ -z "${QOS}" ]]; then
    case "${TIME}" in
        [3-9]-*|[1-9][0-9]-*) QOS="long" ;;
    esac
fi

# Pass training-side env vars through to run_training.sh (post-train hooks).
EXPORT_VARS="ALL"
EXPORT_VARS+=",TRAIN_CMD=${TRAIN_CMD}"
EXPORT_VARS+=",JOB_OUT_DIR=${JOB_OUT_DIR},OUT_DIR=${JOB_OUT_DIR}"
EXPORT_VARS+=",DOMAIN=${DOMAIN},AECLASS=${AECLASS}"
EXPORT_VARS+=",U=${U},A=${A},P=${P}"
EXPORT_VARS+=",CATEGORY=${CATEGORY}"
EXPORT_VARS+=",NPZ_PATH=${NPZ_PATH}"
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
_CAT_DISP="<n/a>"
[[ "${DOMAIN}" =~ ^(vidvrd|actiongenome)$ ]] && _CAT_DISP="${CATEGORY:-_all}"
echo "[submit] Domain   : ${DOMAIN}  Category: ${_CAT_DISP}"
echo "[submit] OUT_DIR  : ${JOB_OUT_DIR}"
[[ "${NPZ_PATH}" != "None" ]] && echo "[submit] NPZ_PATH : ${NPZ_PATH}"
echo "[submit] Cmd      : ${TRAIN_CMD}"
echo "[submit] Hooks    : EXTRACT_FOL=${EXTRACT_FOL:-1}  VISUALIZE=${VISUALIZE:-1}"
echo "[submit] Logs     : ${PROJECT_DIR}/logs/${JOB_NAME}.<JOBID>.{out,err}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[submit] DRY_RUN=1 — would run:"
    echo "  sbatch ${SBATCH_ARGS[*]} ${PROJECT_DIR}/run_training.sh"
    exit 0
fi

sbatch "${SBATCH_ARGS[@]}" "${PROJECT_DIR}/run_training.sh"
