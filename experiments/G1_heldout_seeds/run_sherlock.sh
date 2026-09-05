#!/usr/bin/env bash
# G1 — held-out clips, and three seeds per arm.  See README.md.
#
# Two gaps close together:
#
#   1. Every FOSAE model so far trained on the 88 screened clips and was then
#      scored on those same 88 clips. This trains on 70 and scores on 18 the
#      model never saw.
#   2. Every headline rests on one training run. This trains three.
#
# Three bakes, not two. The obvious comparison is train70 against test18 and it
# is the wrong one: latplan/puzzles/util.py runs equalize_hist over the whole
# array and then rescales by the global minimum and maximum, so a 70-clip bake
# and an 18-clip bake map the same pixel to different values. seen18 holds 18
# clips the model DID train on, baked on its own, so the preprocessing sample
# size matches test18 and the only remaining difference is seen against unseen.
#
# One command, then walk away. The export chains itself on with --dependency:
#
#     cd $SCRATCH/panos/sgg-thesis && git pull
#     mkdir -p logs && sbatch experiments/G1_heldout_seeds/run_sherlock.sh
#
# Then, once the queue empties:
#
#     git add -f eval/exports/G1-seed*.npz && git commit -m "G1 exports" && git push
#     # and on the workstation
#     git pull && bash experiments/G1_heldout_seeds/score_local.sh
#
# The wall clock covers the BAKE, not the training. The three arms are separate
# jobs with their own budgets. A killed bake aborts before any `submit` runs,
# which loses the whole one-visit cycle rather than part of it, so the budget
# here is generous: E1 baked 22 clips inside a 4 h ceiling and this bakes 106.
#SBATCH --job-name=fosae-G1
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/G1.%j.out
#SBATCH --error=logs/G1.%j.err

set -eo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p logs

source venv/bin/activate 2>/dev/null || source activate.sh
source sh/sweep_lib.sh

FPS="${FPS:-30}"
NPZ="data/npz/video/vidvrd/overfit"

TRAIN_STEM="G1-train70-${FPS}fps-mo3-nofill-p8"
TEST_STEM="G1-test18-${FPS}fps-mo3-nofill-p8"
SEEN_STEM="G1-seen18-${FPS}fps-mo3-nofill-p8"

# ── the split, pinned here ──────────────────────────────────────────────────
#
# eval/ is excluded from version control, so eval/vidvrd_winnable_clips.txt
# never reaches Sherlock and reading it here would abort the run. sh/h14.sh
# pins its clip list for exactly that reason, after the defect cost one cycle.
#
# Produced by, and re-derivable with:
#
#   python3 - <<'PY'
#   import hashlib
#   ids = [l.strip() for l in open('eval/vidvrd_winnable_clips.txt') if l.strip()]
#   key = lambda c: hashlib.sha1(("G1-heldout-2026-09-05:" + c).encode("utf-8")).hexdigest()
#   order = sorted(ids, key=key)
#   print("test18 ", ",".join(order[:18]))
#   print("train70", ",".join(order[18:]))
#   print("seen18 ", ",".join(order[18:36]))
#   PY
#
# A sha1 of the clip id under a fixed seed string, not a shuffle: it depends on
# no random number generator and no Python version, so the split is the same
# everywhere and forever. Verified 2026-09-05 to reproduce the three lists
# below, and the 88 input ids match eval/vidvrd_winnable_clips.txt in set and
# in order.
SPLIT_SEED="G1-heldout-2026-09-05"

TRAIN_IDS="$(echo "\
    ILSVRC2015_train_00016000,ILSVRC2015_train_00119037,ILSVRC2015_train_00415004,ILSVRC2015_train_00149006,
    ILSVRC2015_train_00300009,ILSVRC2015_train_00574002,ILSVRC2015_train_00773000,ILSVRC2015_train_00253030,
    ILSVRC2015_train_00010024,ILSVRC2015_train_00897007,ILSVRC2015_train_00025022,ILSVRC2015_train_00118005,
    ILSVRC2015_train_00033006,ILSVRC2015_train_00100002,ILSVRC2015_train_00040001,ILSVRC2015_train_00804001,
    ILSVRC2015_train_00150024,ILSVRC2015_train_00127000,ILSVRC2015_train_00040029,ILSVRC2015_train_00218002,
    ILSVRC2015_train_00065002,ILSVRC2015_train_00290020,ILSVRC2015_train_00535000,ILSVRC2015_train_00211004,
    ILSVRC2015_train_00234013,ILSVRC2015_train_00265048,ILSVRC2015_val_00026002,ILSVRC2015_val_00037004,
    ILSVRC2015_train_00375001,ILSVRC2015_train_00312007,ILSVRC2015_train_00119025,ILSVRC2015_train_00010006,
    ILSVRC2015_train_00040020,ILSVRC2015_train_00068002,ILSVRC2015_train_00057003,ILSVRC2015_train_00040025,
    ILSVRC2015_train_00987000,ILSVRC2015_train_00897009,ILSVRC2015_train_00069006,ILSVRC2015_train_00308005,
    ILSVRC2015_val_00036008,ILSVRC2015_val_00159002,ILSVRC2015_train_00772005,ILSVRC2015_train_00119014,
    ILSVRC2015_train_00071019,ILSVRC2015_train_00165000,ILSVRC2015_train_00415008,ILSVRC2015_train_00265004,
    ILSVRC2015_train_00272001,ILSVRC2015_val_00035008,ILSVRC2015_train_00040031,ILSVRC2015_train_01052000,
    ILSVRC2015_train_00962007,ILSVRC2015_train_00040009,ILSVRC2015_train_00150010,ILSVRC2015_train_00234021,
    ILSVRC2015_train_00211000,ILSVRC2015_train_00181011,ILSVRC2015_train_00308009,ILSVRC2015_train_00119040,
    ILSVRC2015_train_00071012,ILSVRC2015_val_00015001,ILSVRC2015_train_01020000,ILSVRC2015_train_00010010,
    ILSVRC2015_train_00194008,ILSVRC2015_train_00010012,ILSVRC2015_train_01081000,ILSVRC2015_train_00040022,
    ILSVRC2015_train_00797000,ILSVRC2015_train_00411000" | tr -d ' \n')"

TEST_IDS="$(echo "\
    ILSVRC2015_train_00058003,ILSVRC2015_train_00058001,ILSVRC2015_train_01081001,ILSVRC2015_train_00077001,
    ILSVRC2015_train_00119045,ILSVRC2015_train_00324000,ILSVRC2015_train_00265008,ILSVRC2015_val_00028003,
    ILSVRC2015_val_00081000,ILSVRC2015_train_00729000,ILSVRC2015_train_00415006,ILSVRC2015_train_00185001,
    ILSVRC2015_train_00466000,ILSVRC2015_train_00165011,ILSVRC2015_train_00010009,ILSVRC2015_train_00548000,
    ILSVRC2015_train_00040018,ILSVRC2015_train_00040005" | tr -d ' \n')"

# The first 18 of the training order. A subset of TRAIN_IDS, so the model saw
# every one of them.
SEEN_IDS="$(echo "\
    ILSVRC2015_train_00016000,ILSVRC2015_train_00119037,ILSVRC2015_train_00415004,ILSVRC2015_train_00149006,
    ILSVRC2015_train_00300009,ILSVRC2015_train_00574002,ILSVRC2015_train_00773000,ILSVRC2015_train_00253030,
    ILSVRC2015_train_00010024,ILSVRC2015_train_00897007,ILSVRC2015_train_00025022,ILSVRC2015_train_00118005,
    ILSVRC2015_train_00033006,ILSVRC2015_train_00100002,ILSVRC2015_train_00040001,ILSVRC2015_train_00804001,
    ILSVRC2015_train_00150024,ILSVRC2015_train_00127000" | tr -d ' \n')"

# ── the invariant that decides whether this experiment means anything ───────
#
# One clip in both lists makes the held-out arm partly in-sample, and nothing
# downstream would notice. Check it before spending a single minute of compute.
list_of () { echo "$1" | tr ',' '\n' | grep -v '^[[:space:]]*$' | sort -u; }

N_TRAIN="$(list_of "${TRAIN_IDS}" | grep -c .)"
N_TEST="$(list_of "${TEST_IDS}" | grep -c .)"
N_SEEN="$(list_of "${SEEN_IDS}" | grep -c .)"

if [[ "${N_TRAIN}" != "70" || "${N_TEST}" != "18" || "${N_SEEN}" != "18" ]]; then
    echo "FATAL: split sizes are ${N_TRAIN}/${N_TEST}/${N_SEEN}, expected 70/18/18." >&2
    exit 2
fi

OVERLAP="$(comm -12 <(list_of "${TRAIN_IDS}") <(list_of "${TEST_IDS}") | grep -c . || true)"
if [[ "${OVERLAP}" != "0" ]]; then
    echo "FATAL: ${OVERLAP} clip(s) appear in BOTH the train and the test list." >&2
    echo "       The held-out arm would be partly in-sample. Nothing was baked." >&2
    exit 2
fi

NOT_SEEN="$(comm -23 <(list_of "${SEEN_IDS}") <(list_of "${TRAIN_IDS}") | grep -c . || true)"
if [[ "${NOT_SEEN}" != "0" ]]; then
    echo "FATAL: ${NOT_SEEN} clip(s) in the seen18 control are NOT in the train list." >&2
    echo "       The in-sample arm would be partly held out. Nothing was baked." >&2
    exit 2
fi

echo "split seed ${SPLIT_SEED}"
echo "train ${N_TRAIN}, held out ${N_TEST}, in-sample control ${N_SEEN}, no overlap"

# ── bake ────────────────────────────────────────────────────────────────────
#
# --fill-annotations is ABSENT and that is deliberate. Every clip here is fully
# annotated, and filling would fabricate transitions (SPEC B35, V24).
section "G1  bake three splits, fill OFF"

bake_split () {
    local stem="$1" ids="$2" want="$3"
    if [[ -f "${NPZ}/${stem}.npz" ]]; then
        echo "have  ${stem}"
        return 0
    fi
    echo "bake  ${stem}  (${want} clips, no fill)"
    local log="logs/G1-bake-${stem}.$$.log"
    python3 setup-dataset.py video_vidvrd all \
        --video-id "${ids}" --fps "${FPS}" \
        --max-objects 3 --patch-size 8 --out-name "${stem}" \
        2>&1 | tee "${log}"

    # puzzle_vidvrd skips a clip whose frames are missing and carries on, so a
    # partial frame extraction would shrink a split in silence while the stem
    # still claims 70 or 18. Fail loudly instead.
    local loaded
    loaded="$(grep -oE '[0-9]+ videos? loaded' "${log}" | tail -1 \
              | grep -oE '^[0-9]+' || true)"
    if [[ -n "${loaded}" && "${loaded}" != "${want}" ]]; then
        echo "FATAL: baked ${loaded} clips into ${stem}, expected ${want}." >&2
        echo "       Frames are missing on this machine. See ${log}." >&2
        exit 3
    fi
    NBAKED=$((NBAKED + 1))
}

bake_split "${TRAIN_STEM}" "${TRAIN_IDS}" "${N_TRAIN}"
bake_split "${TEST_STEM}"  "${TEST_IDS}"  "${N_TEST}"
bake_split "${SEEN_STEM}"  "${SEEN_IDS}"  "${N_SEEN}"

# ── train ───────────────────────────────────────────────────────────────────
section "G1  train three seeds of one configuration on the 70-clip split"

# The configuration that learned (val 0.1216), at the latent size measured to
# plan: U40 A2 P10, 400 bits. Nothing here is swept. The only thing that varies
# between the three jobs is the output tree, and therefore the initialisation
# TensorFlow picks for itself.
SWEEP_DEFAULTS=(EPOCH=3000 LR=0.001 BATCH=1000
                PREENC_LAYERS=2 PREENC_DIM=1000
                MAX_TEMPERATURE=1.0 TRANSITION_MODE=sequential
                U=40 A=2 P=10)

# A separate OUT_DIR per seed, and this is load-bearing twice over.
#   1. sh/submit.sh composes JOB_OUT_DIR from a hash of the parameters, and
#      three identical configurations hash the same. All three jobs would write
#      one net0.h5 and one training_history.csv.
#   2. latplan/util/paths.py records the other half: each tree needs its own
#      grid_search.log, because a shared log plus LIMIT=1 makes the later runs
#      short-circuit and report the first run's result as their own.
MEM="${MEM:-48G}"
TIME="${TIME:-10:00:00}"

for SEED in 1 2 3; do
    submit "G1 seed ${SEED}  (U40 A2 P10, 400 bits, 70 clips)" \
           "${TRAIN_STEM}" "${MEM}" "${TIME}" \
           OUT_DIR="${PROJECT_DIR}/out/G1/seed${SEED}"
done

sweep_totals

# ── chain the export, so this is a one-visit experiment ─────────────────────
if (( ${#SUBMITTED_IDS[@]} == 0 )); then
    echo "nothing submitted, so nothing to export" >&2
    exit 1
fi

DEP="$(IFS=:; echo "${SUBMITTED_IDS[*]}")"
echo
echo "chaining export after ${#SUBMITTED_IDS[@]} training job(s): ${DEP}"
# afterany, not afterok: a seed that dies should not withhold the seeds that
# lived. export_sherlock.sh exports whatever landed and says what did not.
# The stems travel in the environment, and the child script carries the same
# defaults, so the chain survives either way.
sbatch --dependency="afterany:${DEP}" \
       --export="ALL,G1_TEST_STEM=${TEST_STEM},G1_SEEN_STEM=${SEEN_STEM}" \
       experiments/G1_heldout_seeds/export_sherlock.sh

cat <<'NOTE'

Submitted. Nothing else to do here -- the export runs itself when training
ends. Watch it with:

    squeue -u $USER

When the queue is empty, push the six exports and score them on the
workstation:

    git add -f eval/exports/G1-seed*.npz && git commit -m "G1 exports" && git push
    # then on the workstation
    git pull && bash experiments/G1_heldout_seeds/score_local.sh
NOTE
