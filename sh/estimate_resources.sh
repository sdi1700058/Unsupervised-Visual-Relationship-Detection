#!/usr/bin/env bash
# estimate_resources.sh — Per Sherlock docs (running-jobs/#estimating-resources):
# right way to size a job is reading `seff <jobid>` / `sacct` of past runs.
# This script automates that for jobs whose name shares a prefix with JOB_NAME,
# and falls back to a vidvrd dataset-size heuristic only when no history exists.
#
# Output:
#   FORMAT=human (default) — readable summary on stderr
#   FORMAT=env             — emits MEM=...; TIME=...; ... lines for `eval`
# Inputs (env): JOB_NAME (required for sacct mode), DOMAIN, CATEGORY, FPS,
#               EPOCH, BATCH, SACCT_DAYS (default 30)

set -eo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DOMAIN="${DOMAIN:-vidvrd}"
CATEGORY="${CATEGORY:-None}"
FPS="${FPS:-3}"
EPOCH="${EPOCH:-5000}"
BATCH="${BATCH:-1000}"
SACCT_DAYS="${SACCT_DAYS:-30}"
FORMAT="${FORMAT:-human}"
JOB_NAME="${JOB_NAME:-}"

EST_MEM="32G"; EST_TIME="12:00:00"; EST_GPUS="1"; EST_CPUS="2"
EST_CONSTRAINT=""; EST_MODE="default"; EST_ROWS=0

emit() {
    if [[ "${FORMAT}" == "env" ]]; then
        echo "MEM=${EST_MEM}; TIME=${EST_TIME}; GPUS=${EST_GPUS}; CPUS=${EST_CPUS};"
        [[ -n "${EST_CONSTRAINT}" ]] && echo "CONSTRAINT=${EST_CONSTRAINT};"
        echo "EST_MODE=${EST_MODE}; EST_ROWS=${EST_ROWS};"
    else
        echo "[estimate] mode=${EST_MODE} rows=${EST_ROWS}" >&2
        echo "[estimate]   MEM=${EST_MEM} TIME=${EST_TIME} GPUS=${EST_GPUS} CPUS=${EST_CPUS} ${EST_CONSTRAINT:+CONSTRAINT=${EST_CONSTRAINT}}" >&2
    fi
}

# ── Mode 1: sacct of past jobs sharing JOB_NAME prefix ──────────────────────
if command -v sacct &>/dev/null && [[ -n "${JOB_NAME}" ]]; then
    # Strip timestamp + PID suffix appended by submit.sh (-YYYYMMDD-HHMMSS-PID).
    PREFIX="$(echo "${JOB_NAME}" | sed -E 's/-[0-9]{8}-[0-9]{6}-[0-9]+$//')"
    START_DATE="$(date -d "${SACCT_DAYS} days ago" +%Y-%m-%d 2>/dev/null \
                  || date -v-"${SACCT_DAYS}"d +%Y-%m-%d 2>/dev/null \
                  || echo "")"
    SACCT_OUT="$(sacct -X -u "${USER}" \
                       ${START_DATE:+--starttime=${START_DATE}} \
                       --name="${PREFIX}" \
                       --state=COMPLETED \
                       --format=Elapsed,MaxRSS -P -n 2>/dev/null || true)"
    # Parse Elapsed=[D-]HH:MM:SS, MaxRSS=NNNK|NNNM|NNNG.
    PARSED="$(echo "${SACCT_OUT}" | awk -F'|' '
        function to_sec(t,   a,b,d,h,m,s) {
            d=0; n=split(t,a,"-"); if(n==2){d=a[1]; t=a[2]} else t=a[1]
            split(t,b,":"); h=b[1]; m=b[2]; s=b[3]
            return ((d*24+h)*60+m)*60+s
        }
        function to_mb(r,   v,u) {
            v=substr(r,1,length(r)-1)+0; u=substr(r,length(r),1)
            if(u=="K") return v/1024
            if(u=="M") return v
            if(u=="G") return v*1024
            return v/1024/1024  # bytes
        }
        $1=="" {next}
        { sec=to_sec($1); if(sec>maxsec) maxsec=sec
          if($2!=""){mb=to_mb($2); if(mb>maxmb) maxmb=mb}
          rows++ }
        END { print rows" "maxsec" "(maxmb==""?0:maxmb) }')"
    EST_ROWS="$(echo "${PARSED}" | awk '{print $1+0}')"
    if (( EST_ROWS > 0 )); then
        MAXSEC="$(echo "${PARSED}" | awk '{print $2+0}')"
        MAXMB="$(echo "${PARSED}"  | awk '{print int($3+0)}')"
        # 25% headroom + 30 min slack.
        REQ_SEC=$(( MAXSEC * 5 / 4 + 1800 ))
        REQ_MB=$((  MAXMB  * 5 / 4 + 2048 ))
        (( REQ_MB < 16384 )) && REQ_MB=16384
        REQ_GB=$(( (REQ_MB + 1023) / 1024 ))
        EST_MEM="${REQ_GB}G"
        # Format time.
        D=$(( REQ_SEC / 86400 )); REM=$(( REQ_SEC % 86400 ))
        H=$(( REM / 3600 )); M=$(( (REM % 3600) / 60 )); S=$(( REM % 60 ))
        if (( D > 0 )); then EST_TIME="$(printf '%d-%02d:%02d:%02d' $D $H $M $S)"
        else                 EST_TIME="$(printf '%02d:%02d:%02d' $H $M $S)"
        fi
        EST_MODE="sacct"
        emit; exit 0
    fi
fi

# ── Mode 2: heuristic from vidvrd dataset size ──────────────────────────────
if [[ "${DOMAIN}" == "vidvrd" ]]; then
    ANN_DIR="${PROJECT_DIR}/data/video/vidvrd/annotations/train"
    FRAMES_DIR="${PROJECT_DIR}/data/video/vidvrd/frames_${FPS}fps/train"
    if [[ -d "${ANN_DIR}" && -d "${FRAMES_DIR}" ]]; then
        CAT_ARG=()
        [[ -n "${CATEGORY}" && "${CATEGORY}" != "None" ]] && CAT_ARG=(--category "${CATEGORY}")
        OUT="$(python3 "${PROJECT_DIR}/inspect_vidvrd.py" \
                --ann-dir "${ANN_DIR}" --frames-dir "${FRAMES_DIR}" --fps "${FPS}" \
                "${CAT_ARG[@]}" 2>/dev/null \
                | grep -E '^(category|ALL CATEGORIES)' | head -1 || true)"
        N_STATES=$(echo "${OUT}" | grep -oE 'frames=[0-9]+' | head -1 | cut -d= -f2)
        N_TRANS=$(echo "${OUT}"  | grep -oE '(sequential_transitions|transitions)=[0-9]+' | head -1 | cut -d= -f2)
        : "${N_STATES:=0}"; : "${N_TRANS:=0}"
        if (( N_STATES > 0 )); then
            PER_STATE_KB=$(( 10 * 3272 * 4 / 1024 ))
            RAW_MB=$(( N_STATES * PER_STATE_KB / 1024 ))
            RAM_GB=$(( RAW_MB / 1024 * 6 + 8 ))
            (( RAM_GB < 16 )) && RAM_GB=16
            (( RAM_GB > 256 )) && RAM_GB=256
            EST_MEM="${RAM_GB}G"
            BATCH_N="${BATCH/None/1000}"
            GPU_MB=$(( BATCH_N * 10 * 3272 * 4 * 50 / 1024 / 1024 ))
            if   (( GPU_MB > 30000 )); then EST_CONSTRAINT="GPU_MEM:80GB"
            elif (( GPU_MB > 14000 )); then EST_CONSTRAINT="GPU_MEM:32GB"
            elif (( GPU_MB > 7000  )); then EST_CONSTRAINT="GPU_MEM:16GB"
            fi
            STEPS=$(( (2 * N_TRANS + BATCH_N - 1) / BATCH_N ))
            (( STEPS < 1 )) && STEPS=1
            SECS=$(( EPOCH * STEPS * 4 / 10 + 1800 ))
            HRS=$(( SECS / 3600 + 1 ))
            if (( HRS > 168 )); then EST_TIME="7-00:00:00"
            elif (( HRS > 48 )); then DAYS=$(( (HRS+23)/24 )); EST_TIME="${DAYS}-00:00:00"
            else (( HRS < 2 )) && HRS=2; EST_TIME="$(printf '%02d:00:00' $HRS)"
            fi
            EST_MODE="heuristic"
        fi
    fi
fi

emit
