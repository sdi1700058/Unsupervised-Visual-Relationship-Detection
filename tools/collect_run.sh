#!/usr/bin/env bash
# tools/collect_run.sh — bundle a training run's eyeballable outputs.
#
# Usage:  bash tools/collect_run.sh <OUT_DIR>
# Result: <OUT_DIR>/_collect.tar.gz
#
# Includes (depth <=2):
#   *.png  *.pdf  *.gv
#   history.json  parameters.json
#   *fol*.json  *.caption.md  loaded_videos.json
#
# scp the tarball to local workstation to eyeball:
#   scp sherlock:<OUT_DIR>/_collect.tar.gz /tmp/

set -eo pipefail
OUT="${1:?usage: $0 <OUT_DIR>}"
[[ -d "$OUT" ]] || { echo "[collect] not a dir: $OUT" >&2; exit 2; }

cd "$OUT"
TAR="_collect.tar.gz"
rm -f "$TAR"

find . -maxdepth 2 \
    \( -name '*.png' -o -name '*.pdf' -o -name '*.gv' \
       -o -name 'history.json' -o -name 'parameters.json' \
       -o -name '*fol*.json' -o -name '*.caption.md' \
       -o -name 'loaded_videos.json' \) -print0 \
    | tar --null -czf "$TAR" --files-from=-

echo "[collect] wrote $OUT/$TAR ($(du -h "$TAR" | cut -f1))"
echo "[collect] contents:"
tar -tzf "$TAR" | sed 's/^/  /'
