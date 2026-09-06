#!/usr/bin/env bash
# M6 - does the same action have the same effect wherever it applies?
#
# Builds the oracle exports the table needs, then measures every row. No GPU,
# no training run, no cluster. Read experiments/M6_effect_determinism/README.md
# before changing a threshold: the readings are pre-registered there.
#
#   bash experiments/M6_effect_determinism/run.sh
#   M6_CLIPS=4 bash experiments/M6_effect_determinism/run.sh   # smoke run
set -u

cd "$(dirname "$0")/../.." || exit 1

# An unbounded local run crashed this workstation on 2026-08-28.
ulimit -v 6000000

PY=.venv-local/bin/python
[ -x "$PY" ] || PY=python3

CLIPS=${M6_CLIPS:-12}
OUT=${M6_OUT:-eval/m6}
EXPORTS=eval/exports/m6
mkdir -p "$EXPORTS" "$OUT"

# ---------------------------------------------------------------------------
# Which clips. The screened lists live under eval/, which is gitignored, so a
# fresh checkout falls back to the first clips in name order. The fallback is
# stated rather than silent, because the two sets are not the same dataset.
# ---------------------------------------------------------------------------
vidvrd_clips() {
    if [ -f eval/vidvrd_winnable_clips.txt ]; then
        grep -v '^#' eval/vidvrd_winnable_clips.txt | grep -v '^$' | head -n "$CLIPS"
    else
        echo "no eval/vidvrd_winnable_clips.txt: using the first clips in name order" >&2
        ls data/video/vidvrd/annotations/train | sed 's/\.json$//' | head -n "$CLIPS"
    fi
}

vidor_clips() {
    if [ -f eval/vidor_winnable_w16.txt ]; then
        grep -v '^#' eval/vidor_winnable_w16.txt | grep -v '^$' \
            | sed 's#^#training/#' | head -n "$CLIPS"
    else
        echo "no eval/vidor_winnable_w16.txt: using the first clips in name order" >&2
        (cd data/video/vidor/annotations && ls validation/*/*.json \
            | sed 's/\.json$//' | head -n "$CLIPS")
    fi
}

# build_oracle <annotation-json> <output-npz> <encoding>
#
# stdout is suppressed and stderr is not. It used to be `>/dev/null 2>&1`, so
# "could not build" was the whole report and a clip lost to a missing reader,
# a malformed annotation or the memory cap all looked the same.
build_oracle() {
    [ -f "$2" ] && return 0
    "$PY" tools/planner/oracle.py "$1" --out "$2" --encoding "$3" \
        --max-objects 3 --no-fill >/dev/null \
        || echo "  could not build $2" >&2
}

echo "building oracle exports for $CLIPS VidVRD clips and $CLIPS VidOR clips"

VIDVRD_ONEHOT=""
VIDVRD_BINARY=""
for clip in $(vidvrd_clips); do
    ann=data/video/vidvrd/annotations/train/$clip.json
    [ -f "$ann" ] || ann=data/video/vidvrd/annotations/test/$clip.json
    [ -f "$ann" ] || { echo "  no annotation for $clip" >&2; continue; }
    build_oracle "$ann" "$EXPORTS/vidvrd-onehot-$clip.npz" onehot
    build_oracle "$ann" "$EXPORTS/vidvrd-binary-$clip.npz" binary
    [ -f "$EXPORTS/vidvrd-onehot-$clip.npz" ] &&
        VIDVRD_ONEHOT="$VIDVRD_ONEHOT,$EXPORTS/vidvrd-onehot-$clip.npz"
    [ -f "$EXPORTS/vidvrd-binary-$clip.npz" ] &&
        VIDVRD_BINARY="$VIDVRD_BINARY,$EXPORTS/vidvrd-binary-$clip.npz"
done

VIDOR_BINARY=""
for rel in $(vidor_clips); do
    ann=data/video/vidor/annotations/$rel.json
    [ -f "$ann" ] || { echo "  no annotation for $rel" >&2; continue; }
    name=$(echo "$rel" | tr '/' '-')
    build_oracle "$ann" "$EXPORTS/vidor-binary-$name.npz" binary
    [ -f "$EXPORTS/vidor-binary-$name.npz" ] &&
        VIDOR_BINARY="$VIDOR_BINARY,$EXPORTS/vidor-binary-$name.npz"
done

# ---------------------------------------------------------------------------
# The rows. A trained export that is not on disk is skipped by name, so a
# missing model costs one row and not the run.
# ---------------------------------------------------------------------------
ROWS=()
add_trained() {   # add_trained <label> <glob>
    for f in $2; do
        [ -f "$f" ] && { ROWS+=("$1=$f"); return 0; }
    done
    echo "  no export matches $2, skipping $1" >&2
}

add_trained "FOSAE U40 P5 VidVRD"  "eval/exports/U40_A2_P5_catH14-winnable88-*.npz"
add_trained "FOSAE U40 P10 VidVRD" "eval/exports/U40_A2_P10_catH14-winnable88-*.npz"
add_trained "FOSAE U40 P20 VidVRD" "eval/exports/U40_A2_P20_catH14-winnable88-*.npz"
add_trained "FOSAE U20 P10 VidVRD" "eval/exports/U20_A2_P10_catH14-winnable88-*.npz"

[ -n "$VIDVRD_ONEHOT" ] && ROWS+=("oracle onehot VidVRD=${VIDVRD_ONEHOT#,}")
[ -n "$VIDVRD_BINARY" ] && ROWS+=("oracle binary VidVRD=${VIDVRD_BINARY#,}")
[ -n "$VIDOR_BINARY" ]  && ROWS+=("oracle binary VidOR=${VIDOR_BINARY#,}")

if [ ${#ROWS[@]} -eq 0 ]; then
    echo "no row could be built. Check eval/exports/ and data/video/." >&2
    exit 2
fi

echo "measuring ${#ROWS[@]} rows and the synthetic positive control"
"$PY" tools/planner/m6_determinism.py "${ROWS[@]}" --positive-control \
    --out-dir "$OUT"
