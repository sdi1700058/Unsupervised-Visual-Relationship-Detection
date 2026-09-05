#!/bin/sh
# M7 -- mean average precision on relation triplets, the metric the field
# reports. Run from the repository root:
#
#     sh experiments/M7_map_triplets/run.sh
#
# Six conditions and one roll-up figure. Read README.md first: it says what
# each outcome means, and it was written before any of these numbers existed.
#
# Nothing here trains a model or touches a GPU. Every condition is given
# ground-truth tubelets, so every one of them sits in VrdONE's
# oracle-trajectory regime (EVAL.md 5.4).
set -e

# The workstation has been crashed once by an unbounded allocation. The probe
# conditions build one feature block of about 75 MB at a time; this cap is
# far above that and far below the machine.
ulimit -v 6000000

PY=.venv-local/bin/python
OUT=eval/M7
M7="$PY tools/planner/m7_map.py"

VIDVRD_TEST='data/video/vidvrd/annotations/test/*.json'
VIDVRD_TRAIN='data/video/vidvrd/annotations/train/*.json'
VIDOR_VAL='data/video/vidor/annotations/validation/*/*.json'
VIDOR_TRAIN='data/video/vidor/annotations/training/*/*.json'

# 1. The harness check. Both of these MUST print 100.00, and every other
#    number below is void if either does not.
$M7 --annotations "$VIDVRD_TEST" --predictor perfect \
    --dataset vidvrd --tag vidvrd-perfect --out-dir $OUT

$M7 --annotations "$VIDOR_VAL" --limit 200 --predictor perfect \
    --dataset vidor --tag vidor-perfect --out-dir $OUT

# 2. The no-vision floor: the training split's most frequent triplets, put on
#    every ordered pair of ground-truth tubelets.
$M7 --annotations "$VIDVRD_TEST" --predictor frequency \
    --prior "$VIDVRD_TRAIN" \
    --dataset vidvrd --tag vidvrd-frequency --out-dir $OUT

$M7 --annotations "$VIDOR_VAL" --limit 200 --predictor frequency \
    --prior "$VIDOR_TRAIN" --prior-limit 800 \
    --dataset vidor --tag vidor-frequency --out-dir $OUT

# 3. The M1 ridge probe read off the oracle latent exports already on disk,
#    cut into 30-frame segments and associated. These are the only conditions
#    that read anything this project produced.
$M7 --annotations "$VIDVRD_TRAIN" "$VIDVRD_TEST" --predictor probe \
    --exports 'eval/probe/batch/*.npz' \
    --dataset vidvrd --tag vidvrd-probe --out-dir $OUT

$M7 --annotations "$VIDOR_TRAIN" --predictor probe \
    --exports 'eval/probe/vidor/*.npz' \
    --dataset vidor --tag vidor-probe --out-dir $OUT

# 4. One axis, with the published ladder drawn on it.
$M7 --compare "$OUT/*/map.json" --out-dir $OUT

$PY -c "import xml.dom.minidom,sys; xml.dom.minidom.parse(sys.argv[1])" \
    $OUT/m7_map.svg
echo "figures parse as XML"
