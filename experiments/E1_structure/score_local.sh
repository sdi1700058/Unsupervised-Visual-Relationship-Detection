#!/usr/bin/env bash
# E1 — score both arms and print the comparison. Runs locally, CPU only.
#
#   bash experiments/E1_structure/score_local.sh
#
# Needs the two exports pulled from Sherlock. Writes eval/planner/E1_summary.md.

set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PY="${PY:-.venv-local/bin/python}"
[[ -x "${PY}" ]] || PY=python3

A=$(ls eval/exports/*E1-structured*.npz 2>/dev/null | head -1 || true)
B=$(ls eval/exports/*E1-unstructured*.npz 2>/dev/null | head -1 || true)

if [[ -z "${A}" || -z "${B}" ]]; then
    cat <<'MISSING' >&2
Both E1 exports are needed and at least one is absent.

Expected something matching:
    eval/exports/*E1-structured*.npz
    eval/exports/*E1-unstructured*.npz

Run experiments/E1_structure/run_sherlock.sh first, export the two models, and
pull them here. See the README.
MISSING
    exit 1
fi

echo "structured   ${A}"
echo "unstructured ${B}"
echo

# Cheap screen first: latent geometry needs no planner and takes seconds. A
# code that does not order frames like the world will not plan, whatever its
# training data, so this can settle the question before any search runs.
echo "=== latent geometry (screen, no planning) ==="
"${PY}" tools/planner/latent_geometry.py "${A}" "${B}" || true
echo

# 6 GB and a short budget: an unbounded search over a wide latent can exhaust
# memory, and this has crashed a workstation before.
ulimit -v 6000000
for pair in "structured:${A}" "unstructured:${B}"; do
    name="${pair%%:*}"; path="${pair#*:}"
    echo "=== planning: ${name} ==="
    bash tools/planner/eval_plannability.sh "${path}" \
        --methods bfs,pddl --window 8 --budget 30 --name "E1-${name}" \
        2>&1 | tail -3
    echo
done

"${PY}" - <<'PYEOF'
import csv, os, statistics as st

def read(name):
    path = "eval/planner/E1-%s/summary.csv" % name
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    ok = [r for r in rows if r["reachability"] == "True"]
    live = [r for r in ok
            if r.get("moving_gt_steps") and int(r["moving_gt_steps"]) >= 6
            and r.get("bbox_mse")]
    return {
        "windows": len(rows), "solved": len(ok), "scored": len(live),
        "mse": st.median([float(r["bbox_mse"]) for r in live]) if live else None,
        "base": st.median([float(r["baseline_mse"]) for r in live]) if live else None,
        "iou": st.median([float(r["bbox_iou"]) for r in live if r["bbox_iou"]])
               if live else None,
        "beats": sum(1 for r in live if r["beats_baseline"] == "True"),
    }

a, b = read("structured"), read("unstructured")
lines = ["# E1 — structure versus no structure", ""]
if not a or not b:
    lines.append("One or both arms produced no summary. Nothing to compare.")
else:
    lines += [
        "| | structured | unstructured |",
        "|---|---|---|",
        "| windows solved | %d/%d | %d/%d |" % (a["solved"], a["windows"],
                                                b["solved"], b["windows"]),
        "| windows with real motion | %d | %d |" % (a["scored"], b["scored"]),
        "| **planner bbox error** | %s | %s |" % (
            "n/a" if a["mse"] is None else "**%.2f**" % a["mse"],
            "n/a" if b["mse"] is None else "**%.2f**" % b["mse"]),
        "| linear baseline | %s | %s |" % (
            "n/a" if a["base"] is None else "%.2f" % a["base"],
            "n/a" if b["base"] is None else "%.2f" % b["base"]),
        "| trajectory IoU | %s | %s |" % (
            "n/a" if a["iou"] is None else "%.3f" % a["iou"],
            "n/a" if b["iou"] is None else "%.3f" % b["iou"]),
        "| beats the straight line | %d | %d |" % (a["beats"], b["beats"]),
        "",
    ]
    if a["mse"] and b["mse"]:
        ratio = b["mse"] / a["mse"]
        lines.append("Unstructured error is **%.2fx** the structured error." % ratio)
        lines.append("")
        # The pre-registered reading. Arm A carries 38% more transitions, so a
        # win below that margin cannot be attributed to structure.
        if ratio >= 1.38:
            lines.append("**Reading: structure predicts plannability.** The "
                         "margin clears the 38% volume confound. Criterion 0 "
                         "is operative; reorganise the dataset search around "
                         "it.")
        elif ratio > 1.0:
            lines.append("**Reading: inconclusive.** Structured wins, but by "
                         "less than the 38% transition-count advantage it "
                         "starts with, so the win cannot be attributed to "
                         "structure.")
        else:
            lines.append("**Reading: structure does not predict plannability "
                         "here.** Criterion 0 is not disproven, but it should "
                         "stop being the organising principle until it is "
                         "tested at the full 74-clip scale.")
    else:
        lines.append("**Reading: no scorable windows in at least one arm.** "
                     "Most likely both models failed to train at ~1,000 "
                     "transitions, which says nothing about Criterion 0. "
                     "Rerun at the full 74-clip, 6,391-transition scale.")

out = "eval/planner/E1_summary.md"
os.makedirs("eval/planner", exist_ok=True)
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nwrote %s" % out)
PYEOF
