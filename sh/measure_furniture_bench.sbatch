#!/usr/bin/env bash
#
# measure_furniture_bench.sbatch — every demonstration of nine FurnitureBench releases
#
#   sbatch --job-name=fb-all sh/measure_furniture_bench.sbatch
#   FB_ONLY=lamp sbatch --job-name=fb-lamp sh/measure_furniture_bench.sbatch
#
# WHY. FurnitureBench's code declares, for each furniture kind, which part
# pairs a finished piece joins (`should_be_assembled`), and the dataset page
# says a reward is 1 at the step a part is assembled. Together those make each
# demonstration a sequence of assembled(i, j) propositions turning true. That
# was read from the code and the page on 2026-09-28 and not from a file. This
# job counts, for every released demonstration, its length and its reward
# steps, and checks the count against the pairs the code declares.
#
# IT STREAMS AND KEEPS NOTHING. The tar.gz is read front to back, each pickle
# is summarised and dropped, and nothing lands on disk but the log.
#
# IT NEVER CALLS pickle.load. The restricted unpickler refuses every global
# outside a numpy allow list, and a test pins that a pickle naming os.system
# is refused.
#
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}" || exit 2
# Every furniture kind at HIGH randomness, the ids of
# furniture_bench/scripts/download_dataset.py read on 2026-09-28. A kind can
# be named alone with FB_ONLY=<kind>, or several with FB_ONLY="<kind> <kind>".
IDS="cabinet:1RHbo27phzXVJDjMXPI91QKPHWByCSQU4
chair:1D8j1s4v9NL02V03PwEU6moOjn9v5qzFn
desk:1xOhzI96-BORgjyqF7rBdYRPc_-RxYr_Z
drawer:1QNqA48y9fFE4251xMmCMaUOdmNcnaZ8T
lamp:1Ia0SHIACIoqwzVjhc_dgzEMsiKffIO6T
one_leg:19iLUPDAvrRzevVggfD09nK2ayZ3GrVfC
round_table:1qS2lIiPdqq8pTJsPN2txU1lzQJ4seWCr
square_table:1Wq1MVCUSXxi6wJk7CQW-3LMGdUGJdiAR
stool:1QeEb4Ajz-qN820Y_DVuxU7UdvrZBv7hJ"
for pair in ${IDS}; do
    kind="${pair%%:*}"
    id="${pair#*:}"
    if [ -n "${FB_ONLY:-}" ]; then
        case " ${FB_ONLY} " in *" ${kind} "*) ;; *) continue ;; esac
    fi
    echo "=== ${kind} high"
    PYTHONPATH="${ROOT}/workbench" "${ROOT}/.venv-local/bin/python" \
        "${ROOT}/workbench/tools/furniture_bench_probe.py" --all \
        "https://drive.usercontent.google.com/download?id=${id}&export=download&confirm=t"
    echo "=== ${kind} exit $?"
    # On 2026-09-28 eight kinds in a row got an HTML page instead of gzip,
    # and the same URLs served gzip minutes later, so the next kind waits.
    sleep 60
done
