#!/usr/bin/env bash
#
# verify_pvsg.sbatch — every PVSG zip against its CRC, and what each holds
#
#   sbatch --dependency=afterok:<fetch job> --job-name=verify-pvsg sh/verify_pvsg.sbatch
#
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${THESIS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
module load python/3.9.0
cd "${ROOT}/workbench" || exit 2
verdict=0
for z in "${ROOT}"/data/video/pvsg/raw/*.zip; do
    PYTHONPATH="..:${PYTHONPATH:-}" python3 -m tools.verify_archive "${z}" || verdict=1
    python3 -c "
import collections, os, sys, zipfile
names = [n for n in zipfile.ZipFile(sys.argv[1]).namelist() if not n.endswith('/')]
print('  by extension', dict(collections.Counter(os.path.splitext(n)[1] for n in names)))
print('  top folders', len({n.split('/')[0] for n in names}), 'second level', len({'/'.join(n.split('/')[:2]) for n in names}))
" "${z}"
done
exit "${verdict}"
