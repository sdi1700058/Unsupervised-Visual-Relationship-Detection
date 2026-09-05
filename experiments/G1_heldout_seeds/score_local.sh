#!/usr/bin/env bash
# G1 — score the six exports on the workstation and say what they support.
#
#     bash experiments/G1_heldout_seeds/score_local.sh
#
# Runs entirely on the local machine, CPU only, and every planner process is
# capped at 6 GB. An unbounded search over a wide latent took this workstation
# down on 2026-08-28.
#
# Resumable. A seed and arm whose summary.csv already exists is skipped, so a
# run that was interrupted after two hours continues where it stopped. Force a
# fresh score with RESCORE=1, and a fresh slice with RESLICE=1.
#
# Knobs:  WINDOW=16  BUDGET=20  MAX_WINDOWS=3  METHODS=bfs,pddl  MEM_KB=6000000

set -eo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PY="${PY:-.venv-local/bin/python}"
[[ -x "${PY}" ]] || PY=python3

# Window 16, not 8. The crossover criterion that selected these 88 clips is a
# steep function of window size: at window 8 the oracle itself loses to a
# straight line on 4 of 4 random screened clips, and at window 16 it wins on 6
# of 10. Scoring at 8 measures something the selection never promised
# (SPEC V37). Override with WINDOW= only with a reason.
WINDOW="${WINDOW:-16}"
BUDGET="${BUDGET:-20}"
MAX_WINDOWS="${MAX_WINDOWS:-3}"
METHODS="${METHODS:-bfs,pddl}"
MEM_KB="${MEM_KB:-6000000}"     # 6 GB ceiling per planner process

SEEDS="${SEEDS:-1 2 3}"
ARMS="seen18 test18"

shopt -s nullglob
PRESENT=(eval/exports/G1-seed*-seen18.npz eval/exports/G1-seed*-test18.npz)
shopt -u nullglob

if (( ${#PRESENT[@]} == 0 )); then
    cat <<'MISSING' >&2
No G1 exports found. Expected files matching:

    eval/exports/G1-seed<N>-seen18.npz
    eval/exports/G1-seed<N>-test18.npz

Run the Sherlock half first:

    cd $SCRATCH/panos/sgg-thesis && git pull
    mkdir -p logs && sbatch experiments/G1_heldout_seeds/run_sherlock.sh

then push what it produced and pull it here. See the README.
MISSING
    exit 1
fi

echo "found ${#PRESENT[@]} of 6 exports:"
printf '  %s\n' "${PRESENT[@]}"
echo

# ── are the three replicates actually different? ────────────────────────────
#
# strips.py carries no seed knob. The three runs differ only because
# TensorFlow picks its own initialiser seed when no graph seed is set. That is
# inferred, not measured, so measure it here: identical latents would mean the
# three runs are one run reported three times, and the seed half of this
# experiment would be void.
echo "=========================================="
echo "stage 1  are the three replicates independent?"
echo "=========================================="
"${PY}" - <<'PYEOF'
import glob
import hashlib
import sys

import numpy as np

digests = {}
for path in sorted(glob.glob("eval/exports/G1-seed*-seen18.npz")):
    data = np.load(path)
    if "latents" not in data.files:
        print("  %s carries no latents" % path)
        continue
    digest = hashlib.sha1(
        np.ascontiguousarray(data["latents"]).tobytes()).hexdigest()[:12]
    digests.setdefault(digest, []).append(path)
    print("  %-44s latents sha1 %s" % (path, digest))

clashes = [group for group in digests.values() if len(group) > 1]
if clashes:
    print("")
    print("  WARNING: these runs encode the same clips to BIT-IDENTICAL")
    print("  latents, so they are not independent replicates and the seed")
    print("  half of G1 measures nothing:")
    for group in clashes:
        for path in group:
            print("    %s" % path)
    print("  Cause to check first: did the three jobs share one OUT_DIR, so")
    print("  the later two short-circuited on a shared grid_search.log?")
elif len(digests) > 1:
    print("")
    print("  %d distinct initialisations. The replicates are independent."
          % len(digests))
sys.exit(0)
PYEOF
echo

# ── the cheap screen, before any planning ───────────────────────────────────
#
# Latent geometry needs no planner and takes seconds. A code that does not
# order frames like the world will not plan, whatever it trained on, so this
# can settle the question before any search runs. It is a screen and not a
# ranking: it correlates with planner error in the WRONG direction across
# resolutions, so it cannot order the arms (SPEC V26).
echo "=========================================="
echo "stage 2  latent geometry screen (seconds)"
echo "=========================================="
mkdir -p eval/planner
"${PY}" tools/planner/latent_geometry.py "${PRESENT[@]}" \
    --csv eval/planner/G1_geometry.csv || true
echo

# ── slice, then plan ────────────────────────────────────────────────────────
#
# Every export holds 18 clips written into one flat frame sequence, with 17
# boundaries inside it. make_windows slides blindly along that axis, so a
# window straddling a boundary pairs the last frame of one video with the first
# frame of another -- a transition that never happened, scored as if it had.
# slice_export.py cuts the export into one file per clip, and
# eval_plannability.sh windows each file on its own.
echo "=========================================="
echo "stage 3  the planner, one arm at a time"
echo "=========================================="
NSCORED=0
NSKIPPED=0
for SEED in ${SEEDS}; do
    for ARM in ${ARMS}; do
        EXPORT="eval/exports/G1-seed${SEED}-${ARM}.npz"
        NAME="G1-seed${SEED}-${ARM}"
        if [[ ! -f "${EXPORT}" ]]; then
            echo "absent  ${EXPORT}"
            NSKIPPED=$((NSKIPPED + 1))
            continue
        fi

        CLIPDIR="eval/exports/g1/seed${SEED}-${ARM}"
        shopt -s nullglob
        SLICES=("${CLIPDIR}"/*.npz)
        shopt -u nullglob
        if (( ${#SLICES[@]} == 0 )) || [[ -n "${RESLICE:-}" ]]; then
            echo "slice   ${EXPORT} -> ${CLIPDIR}"
            "${PY}" tools/planner/slice_export.py "${EXPORT}" \
                --all --out-dir "${CLIPDIR}" --min-frames "${WINDOW}"
        else
            echo "sliced  ${CLIPDIR} (${#SLICES[@]} clips)"
        fi

        if [[ -f "eval/planner/${NAME}/summary.csv" && -z "${RESCORE:-}" ]]; then
            echo "scored  ${NAME} already, skipping. RESCORE=1 to redo it."
            NSCORED=$((NSCORED + 1))
            echo
            continue
        fi

        echo
        echo "--- ${NAME}"
        # `bfs,pddl` and not `bfs` alone: BFS solved 0 of 14 windows on good
        # data, because good annotation makes nearly every transition unique
        # and blind search needs repetition to compose them. That is a
        # statement about the search, not about the representation. The
        # summary credits each window with the better of the two.
        # The output streams rather than passing through `tail`. This arm can
        # take half an hour and a silent terminal looks like a hang.
        if ( ulimit -v "${MEM_KB}"
             bash tools/planner/eval_plannability.sh "${CLIPDIR}" \
                 --methods "${METHODS}" \
                 --window "${WINDOW}" \
                 --max-windows "${MAX_WINDOWS}" \
                 --budget "${BUDGET}" \
                 --length-mode max \
                 --name "${NAME}" ); then
            NSCORED=$((NSCORED + 1))
        else
            echo "  ${NAME} did not complete"
            NSKIPPED=$((NSKIPPED + 1))
        fi
        echo
    done
done

echo "scored ${NSCORED} arm(s), skipped ${NSKIPPED}"
echo

# ── the summary and the picture ─────────────────────────────────────────────
echo "=========================================="
echo "stage 4  the comparison"
echo "=========================================="
"${PY}" experiments/G1_heldout_seeds/g1_summary.py

# The per-arm charts and the written verdict, the way every other planner run
# in this project reports itself. make_report.py needs the standard library
# only, so it works anywhere; viz_plannability.py needs matplotlib and writes
# its own MPLCONFIGDIR, which warns on every call when the directory is absent.
"${PY}" tools/planner/make_report.py eval/planner/G1-seed*/ --index || true

MPLDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mpl}"
mkdir -p "${MPLDIR}"
for D in eval/planner/G1-seed*/; do
    [[ -f "${D}/summary.csv" ]] || continue
    MPLCONFIGDIR="${MPLDIR}" "${PY}" tools/planner/viz_plannability.py "${D}" || true
done

cat <<'NOTE'

Read the comparison at:  eval/planner/G1_summary.md
The picture:             eval/planner/G1_summary.svg
Per-arm verdicts:        eval/planner/index.html

How to read it, and this was fixed before the run:

  drop   = median in-sample solve rate, minus median held-out solve rate
  noise  = the larger of the two seed spreads, highest minus lowest

  drop inside noise    the model plans on clips it never saw as well as on
                       clips it did. The earlier headlines can be restated as
                       held-out numbers.
  drop above noise     part of every earlier score was memorisation. Relabel
                       the earlier headlines as in-sample.
  held out near zero   the operator set does not reach states from unseen
                       clips. A measured negative with a named cause.
  noise as large as
  any arm difference   one run per arm was never enough. Every single-seed
                       headline needs three seeds before it is reported again.
NOTE
