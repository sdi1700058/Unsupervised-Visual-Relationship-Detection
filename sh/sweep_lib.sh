#!/usr/bin/env bash
# Shared bake and submit helpers for the sweep scripts.
#
# Three sweeps grew their own copies of these two functions and the copies
# drifted. The worst of the drift: overnight_sweep.sh put "$@" before its own
# MEM and TIME, so a caller could not override either and the TIME=2:00:00 on
# its line 175 was discarded in silence. Here the caller's variables come last
# and therefore win.
#
# Source it, set the defaults, then call the functions:
#
#     source sh/sweep_lib.sh
#     NPZ="data/npz/video/vidvrd/overfit"
#     FPS=30
#     SWEEP_DEFAULTS=(EPOCH=2000 LR=0.001 PREENC_LAYERS=2 PREENC_DIM=1000)
#     submit "arm A" "${CLIP}" 32G 4:00:00 U=40 P=20
#
# The counters NBAKED, NJOBS and NFAILED are updated in place, so a sweep can
# print a total at the end.

NBAKED="${NBAKED:-0}"
NJOBS="${NJOBS:-0}"
NFAILED="${NFAILED:-0}"

# Every job id `submit` gets back, so a sweep can chain a follow-up step onto
# all of its arms with --dependency instead of needing a second visit.
SUBMITTED_IDS=()

# The environment every arm gets unless it says otherwise. A sweep overwrites
# this array with its own baseline, so an arm only has to name what it changes.
#
# NOT `("${SWEEP_DEFAULTS[@]:-}")`. That idiom looks defensive but yields a
# one-element array holding the EMPTY STRING, and `env` then takes "" as the
# command name and exits 127 -- every arm of such a sweep failing through the
# SUBMIT FAILED path. Measured 2026-08-30.
if [[ -z "${SWEEP_DEFAULTS+x}" ]]; then
    SWEEP_DEFAULTS=()
fi

section () {
    echo
    echo "=========================================="
    echo "$*"
    echo "=========================================="
}

# bake <out_name> <category> [setup-dataset.py args ...]
#
# Skips the build when the npz already exists, because baking is the slow part
# and a sweep is usually rerun to add arms rather than to rebuild data.
bake () {
    local name="$1" cat="$2"; shift 2
    [[ -f "${NPZ}/${name}.npz" ]] && { echo "have  ${name}"; return 0; }
    echo "bake  ${name}"
    if python3 setup-dataset.py video_vidvrd "${cat}" \
           --fps "${FPS}" --out-name "${name}" "$@"; then
        NBAKED=$((NBAKED + 1))
    else
        echo "  BAKE FAILED ${name}"
        NFAILED=$((NFAILED + 1))
    fi
}

# submit <tag> <npz_stem> <mem> <time> [VAR=val ...]
#
# The trailing VAR=val pairs come after SWEEP_DEFAULTS in the env invocation,
# so an arm always overrides the baseline rather than the other way round.
submit () {
    local tag="$1" stem="$2" mem="$3" time="$4"; shift 4
    local f="${NPZ}/${stem}.npz"
    if [[ ! -f "${f}" ]]; then
        echo "SKIP  ${tag}  (no ${stem}.npz)"
        return 0
    fi
    echo "--- ${tag}"
    local out
    # Order is load-bearing, and this is the second time it has bitten. Later
    # assignments win in `env`, so the precedence must read
    #   SWEEP_DEFAULTS  <  this call's MEM/TIME  <  the arm's own VAR=val
    # MEM/TIME used to come FIRST, so a sweep that put MEM or TIME in its
    # defaults silently overrode the per-arm resource request -- exactly the
    # regression this file was extracted to eliminate, just relocated.
    # A wrong TIME on Sherlock costs a night. Measured 2026-08-30.
    if out="$(env NPZ_PATH="${PWD}/${f}" FPS="${FPS}" \
                  DOMAIN="${DOMAIN:-vidvrd}" NO_EARLYSTOP=1 AUTO_RESOURCES=0 \
                  "${SWEEP_DEFAULTS[@]}" \
                  MEM="${mem}" TIME="${time}" \
                  "$@" \
                  bash sh/submit.sh 2>&1)"; then
        echo "${out}" | grep -E "Submitted|OUT_DIR" || echo "${out}" | tail -2
        local id
        id="$(echo "${out}" | sed -n 's/^Submitted batch job \([0-9]\+\).*/\1/p' | tail -1)"
        [[ -n "${id}" ]] && SUBMITTED_IDS+=("${id}")
        NJOBS=$((NJOBS + 1))
    else
        # A silent failure here costs a whole arm, so show why.
        echo "  SUBMIT FAILED:"
        echo "${out}" | tail -5 | sed 's/^/    /'
        NFAILED=$((NFAILED + 1))
    fi
}

sweep_totals () {
    echo
    echo "baked ${NBAKED}, submitted ${NJOBS}, failed ${NFAILED}"
}
