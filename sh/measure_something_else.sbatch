#!/usr/bin/env bash
#
# measure_something_else.sbatch — the human-box half of the Something-Else row.
#
#   sbatch --job-name=se-measure sh/measure_something_else.sbatch
#
# WHY A JOB. Three box files at about 1.1 GB each, and a login-node attempt on
# 2026-09-26 produced no output at all before it was abandoned. The PDDL half
# runs on a login node in under a second and is not repeated here.
#
# WHAT THIS ROW IS FOR. Two adversaries observed that this project's two
# highest change rates, Endoscapes 67.5 per cent and Ego4D-EASG 70.1, both
# come from a predicate rule THIS PROJECT wrote. Something-Else is the only
# candidate whose rule somebody else published: Migimatsu and Bohg, ICRA 2022,
# arXiv 2109.14718, hand-author a PDDL domain over all 174 SSv2 action
# templates and take their boxes from Something-Else.
#
# MEASURED ON THE LOGIN NODE ALREADY, from env/twentybn/domain.pddl: 37
# predicates of which 10 are object-to-object, 174 actions, and 67 of those
# actions (38.5 per cent) flip a binary predicate. That is a change rate the
# published domain asserts, with no threshold of this project's in it.
#
# PART ONE IS MISSING. `bounding_box_smthsmth_part1.json` answers HTTP 500 on
# every attempt from that Google Drive id, so this runs over parts 2, 3 and 4.
# The `download` file beside them is a byte-identical duplicate of part 2 from
# the filename collision that was fixed the same day, and it is NOT read here.
#
#SBATCH --partition=normal
#SBATCH --time=04:00:00
# `normal` caps memory at 8 GB per core, so 64 GB needs 8 cores. Measured
# with sh_part on 2026-09-26. An attempt at bigmem was refused: that
# partition demands a 256 GB minimum, which this job does not need.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${ROOT}" || exit 2
module load python/3.12.1

RAW="${ROOT}/data/video/something_else/raw"
PYTHONPATH="${ROOT}/workbench" python3 "${ROOT}/workbench/tools/something_else_relations.py" \
    --domain "${RAW}/twentybn_domain.pddl" \
    --boxes "${RAW}/bounding_box_smthsmth_part2.json" \
            "${RAW}/bounding_box_smthsmth_part3.json" \
            "${RAW}/bounding_box_smthsmth_part4.json"
