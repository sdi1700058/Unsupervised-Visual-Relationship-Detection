#!/usr/bin/env bash
# Export every G1 seed that finished, twice: once on the clips it trained on
# and once on the clips it never saw. Chained off run_sherlock.sh by
# --dependency, so it is not normally submitted by hand.
#
# Why this exists instead of sh/export_model.sh:
#
#   1. sh/export_model.sh always encodes the npz the model trained on. The
#      held-out half of G1 needs `--npz-path`, which points the SAME trained
#      model at a DIFFERENT bake. export_latents.py has accepted that flag all
#      along; nothing else in the project has used it.
#   2. sh/export_model.sh derives the output name from the model directory, so
#      the two encodings of one model would overwrite each other.
#
# The planner operators come from actions.csv in the model directory, which
# holds the TRAINING transitions. They are identical in both arms, so a
# held-out window asks the right question: can operators learned on 70 clips
# reach a goal taken from a clip the model never saw?
#
# The glob has to be evaluated HERE rather than in run_sherlock.sh, because the
# run directories do not exist until the training jobs have started.
#
# Export cannot run on a login node: TensorFlow fails to build its thread pool
# against the login node's process limit and aborts. See sh/export_model.sh.
#
#SBATCH --job-name=fosae-G1-export
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/G1-export.%j.out
#SBATCH --error=logs/G1-export.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs eval/exports

source venv/bin/activate 2>/dev/null || source activate.sh

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"

# Passed in by run_sherlock.sh. The defaults repeat its own, so a hand-run of
# this script still works and a broken --export still lands on the right file.
TEST_STEM="${G1_TEST_STEM:-G1-test18-${FPS}fps-mo3-nofill-p8}"
SEEN_STEM="${G1_SEEN_STEM:-G1-seen18-${FPS}fps-mo3-nofill-p8}"

for STEM in "${SEEN_STEM}" "${TEST_STEM}"; do
    if [[ ! -f "${NPZ}/${STEM}.npz" ]]; then
        echo "FATAL: ${NPZ}/${STEM}.npz is absent, so nothing can be encoded." >&2
        echo "       Did the bake in run_sherlock.sh finish, and do the stems" >&2
        echo "       in the two scripts agree? Override with G1_SEEN_STEM= and" >&2
        echo "       G1_TEST_STEM=." >&2
        exit 2
    fi
done

# No GPU is needed to encode a couple of thousand frames, and asking for one
# only lengthens the queue wait.
export FOSAE_GPU=0
export CUDA_VISIBLE_DEVICES=""

shopt -s nullglob
DIRS=(out/G1/seed*/video/vidvrd/*G1-train70*)
shopt -u nullglob

if (( ${#DIRS[@]} == 0 )); then
    echo "no G1 run directories under out/G1/seed*/video/vidvrd/ -- every seed failed" >&2
    exit 1
fi

echo "found ${#DIRS[@]} G1 run directory(ies)"

rc=0
DONE_SEEDS=""
for MODEL_DIR in "${DIRS[@]}"; do
    SEED="$(echo "${MODEL_DIR}" | sed -n 's#^out/G1/seed\([0-9][0-9]*\)/.*#\1#p')"
    if [[ -z "${SEED}" ]]; then
        echo "SKIP ${MODEL_DIR}: no seed number in the path"
        rc=1
        continue
    fi

    # latplan/util/paths.py appends _2, _3 when a directory already exists, so
    # one seed can hold more than one run after a resubmission. Take the first
    # and say what was left, rather than letting the second overwrite the
    # first's export in silence.
    case " ${DONE_SEEDS} " in
        *" ${SEED} "*)
            echo "SKIP ${MODEL_DIR}: seed ${SEED} is already exported from an"
            echo "     earlier run directory. Export it by hand if you want it."
            continue
            ;;
    esac
    DONE_SEEDS="${DONE_SEEDS} ${SEED}"

    echo
    echo "=== seed ${SEED}  ${MODEL_DIR}"
    for PAIR in "seen18:${SEEN_STEM}" "test18:${TEST_STEM}"; do
        ARM="${PAIR%%:*}"
        STEM="${PAIR#*:}"
        OUT="eval/exports/G1-seed${SEED}-${ARM}.npz"
        echo "--- ${ARM} -> ${OUT}"
        if python3 tools/planner/export_latents.py "${MODEL_DIR}" \
               --npz-path "${NPZ}/${STEM}.npz" -o "${OUT}"; then
            ls -lh "${OUT}"
        else
            echo "FAILED ${ARM} for seed ${SEED}"
            rc=1
        fi
    done
done

echo
echo "exports on disk:"
ls -lh eval/exports/G1-seed*.npz 2>/dev/null || echo "  none"

cat <<'NOTE'

Push what landed, then score it on the workstation:

    git add -f eval/exports/G1-seed*.npz && git commit -m "G1 exports" && git push
    # on the workstation
    git pull && bash experiments/G1_heldout_seeds/score_local.sh
NOTE

exit "${rc}"
