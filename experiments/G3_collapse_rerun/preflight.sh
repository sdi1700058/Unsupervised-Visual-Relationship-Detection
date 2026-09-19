#!/usr/bin/env bash
# preflight.sh - everything the re-run needs, checked in seconds on a login
# node, before a night is spent finding out.
#
#     bash experiments/G3_collapse_rerun/preflight.sh
#
# It writes nothing and submits nothing. A FAIL line names what to do.

set -uo pipefail

SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SH_DIR}" && until [ -f strips.py ] || [ "${PWD}" = / ]; do cd ..; done; pwd)"
PRIV="${ROOT}/workbench/sh"

fails=0
ok   () { printf '  ok    %s\n' "$*"; }
fail () { printf '  FAIL  %s\n' "$*"; fails=$((fails + 1)); }

echo "public root: ${ROOT}"
[ -f "${ROOT}/strips.py" ] && ok "strips.py" || fail "no public root found"

echo
echo "the runners"
for f in submit.sh sweep_lib.sh h14_export.sh export_model.sh estimate_resources.sh; do
    [ -f "${PRIV}/${f}" ] && ok "${f}" || fail "missing ${PRIV}/${f}"
done
for f in submit.sh sweep_lib.sh h14_export.sh export_model.sh; do
    bash -n "${PRIV}/${f}" 2>/dev/null && ok "${f} parses" || fail "${f} has a syntax error"
done

echo
echo "the baked dataset"
NPZ="${ROOT}/data/npz/video/vidvrd/overfit/H14-winnable88-30fps-mo3-nofill-p8.npz"
if [ -f "${NPZ}" ]; then
    ok "$(basename "${NPZ}")  $(du -h "${NPZ}" | cut -f1)"
else
    fail "no ${NPZ}"
    echo "        The re-run does not bake. Run workbench/sh/h14.sh instead,"
    echo "        which bakes and then trains all four of its own arms."
fi

echo
echo "the python environment"
if [ -f "${ROOT}/venv/bin/activate" ]; then
    ok "venv/bin/activate"
else
    fail "no ${ROOT}/venv/bin/activate"
fi

echo
echo "slurm"
command -v sbatch >/dev/null && ok "sbatch on PATH" || fail "no sbatch: this is not a Sherlock shell"
command -v squeue >/dev/null && ok "squeue on PATH" || fail "no squeue"

echo
echo "the five collapsed runs, which this re-run overwrites in place"
for d in U20_A2_P10 U40_A2_P20 U10_A2_P10 U20_A2_P20 U5_A2_P5; do
    hit="$(ls -d "${ROOT}"/out/video/vidvrd/FirstOrderSAE_${d}_catH14-winnable88-* 2>/dev/null | head -1)"
    if [ -n "${hit}" ]; then
        ok "${d}  ->  $(basename "${hit}")"
    else
        printf '  note  %s has no run directory yet, so it trains fresh\n' "${d}"
    fi
done

echo
echo "disk"
df -h "${ROOT}" | tail -1 | awk '{printf "  %s used of %s, %s free\n", $3, $2, $4}'

echo
if [ "${fails}" -eq 0 ]; then
    echo "preflight clean. Submit with:"
    echo "    cd ${ROOT} && mkdir -p logs && sbatch experiments/G3_collapse_rerun/rerun_collapsed.sh"
    exit 0
fi
echo "${fails} check(s) failed. Nothing was submitted."
exit 1
