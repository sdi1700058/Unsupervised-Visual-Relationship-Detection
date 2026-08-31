#!/usr/bin/env python3
"""Summarise the two arms of E1 and state what the comparison supports.

**Why this is a module and not a heredoc.** The first version lived inline in
`experiments/E1_structure/score_local.sh` and it reported the opposite of what
the data said. Structured solved 160 of 160 windows; unstructured solved 8 of
116. The summary then compared the two median errors as though they described
comparable samples, found unstructured lower, and printed *"structure does not
predict plannability"*. The 8 windows were the only ones the unstructured
planner could reach at all, so they are the easiest 7% of its corpus, measured
against the whole of the other arm.

Two rules come out of that, and they are what this module enforces:

1. **Compare `mse_ratio`, never raw `bbox_mse`, across arms.** The two arms run
   on different clips. `SPEC.md` V38 measured that raw planner error is
   dominated by how non-linear a clip happens to be, and the linear baseline
   spans 1575x across clips where the oracle's error spans 5x. A raw-error
   comparison across different clips therefore measures the clips.
2. **Solve rate is a result, not a filter.** An arm that reaches 7% of its
   windows has already lost, and the errors on the 7% it did reach are
   optimistically biased by selection. Report the rate first and say plainly
   that the error column understates the gap.

Python 3.6 clean, standard library only.
"""

import csv
import os
import statistics as st


# Arm A carries 38% more transitions than arm B. Any advantage smaller than
# that could be volume rather than structure, so it does not count as support.
VOLUME_CONFOUND = 1.38

# Below this fraction of windows reached, an arm's error median describes a
# self-selected sample and must be labelled as such.
SELECTION_FLOOR = 0.5


def read(name, root="eval/planner"):
    """One arm's numbers, or None when the run is absent."""
    path = os.path.join(root, "E1-%s" % name, "summary.csv")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    return summarise_rows(rows)


def summarise_rows(rows):
    """The numbers one arm contributes, from its raw summary rows."""
    solved = [r for r in rows if (r.get("reachability") or "").strip().lower()
              == "true"]
    live = [r for r in solved
            if r.get("moving_gt_steps") and float(r["moving_gt_steps"]) >= 6
            and r.get("bbox_mse")]
    ratios = [float(r["mse_ratio"]) for r in live
              if (r.get("mse_ratio") or "").strip() not in ("", "None")]
    return {
        "windows": len(rows),
        "solved": len(solved),
        "scored": len(live),
        "solve_rate": (float(len(solved)) / len(rows)) if rows else 0.0,
        "ratio": st.median(ratios) if ratios else None,
        "mse": st.median([float(r["bbox_mse"]) for r in live]) if live else None,
        "base": st.median([float(r["baseline_mse"]) for r in live])
                if live else None,
        "iou": st.median([float(r["bbox_iou"]) for r in live if r["bbox_iou"]])
               if live else None,
        "beats": sum(1 for r in live if r.get("beats_baseline") == "True"),
    }


def reading(a, b):
    """What the two arms support, as (verdict, list of caveats).

    `a` is the structured arm and `b` the unstructured one.
    """
    caveats = []
    if a is None or b is None:
        return ("**No reading.** One arm produced no summary.", caveats)
    if not a["scored"] or not b["scored"]:
        return ("**Reading: no scorable windows in at least one arm.** Most "
                "likely a model did not train at this scale, which says "
                "nothing about Criterion 0. Rerun at the full 88-clip, "
                "8,522-transition scale.", caveats)

    # Solve rate first. It is the axis least confounded by clip difficulty,
    # because reaching the goal at all does not depend on how far the boxes
    # move.
    rate_gain = (a["solve_rate"] / b["solve_rate"]) if b["solve_rate"] else None
    if rate_gain is None or rate_gain > VOLUME_CONFOUND:
        verdict = ("**Reading: structure predicts plannability, on reachability.**"
                   " The structured arm reaches %.0f%% of its windows against "
                   "%.0f%% for the unstructured one%s. Reaching a goal does not "
                   "depend on how far the boxes move, so this axis is not the "
                   "one V38 warns about."
                   % (100 * a["solve_rate"], 100 * b["solve_rate"],
                      "" if rate_gain is None else ", a factor of %.1f" % rate_gain))
    elif a["solve_rate"] > b["solve_rate"]:
        verdict = ("**Reading: inconclusive on reachability.** The structured "
                   "arm reaches more windows, but by less than the %.0f%% "
                   "transition-count advantage it starts with."
                   % (100 * (VOLUME_CONFOUND - 1)))
    else:
        verdict = ("**Reading: structure does not predict reachability here.** "
                   "Criterion 0 is not disproven, but it should stop being the "
                   "organising principle until it is tested at the full "
                   "88-clip scale.")

    # Error, and only as a ratio. Raw error across different clips measures the
    # clips (V38).
    if a["ratio"] and b["ratio"]:
        gain = b["ratio"] / a["ratio"]
        if gain >= VOLUME_CONFOUND:
            caveats.append("On `mse_ratio` the structured arm is **%.2fx** "
                           "better, which clears the volume confound." % gain)
        elif gain > 1.0:
            caveats.append("On `mse_ratio` the structured arm is %.2fx better, "
                           "which does **not** clear the %.2fx volume "
                           "confound. That axis is inconclusive."
                           % (gain, VOLUME_CONFOUND))
        else:
            caveats.append("On `mse_ratio` the unstructured arm is %.2fx "
                           "better." % (1.0 / gain))

    # Selection bias, whenever an arm reached only part of its corpus.
    for label, arm in (("structured", a), ("unstructured", b)):
        if arm["solve_rate"] < SELECTION_FLOOR:
            caveats.append("The %s error figures cover only the %d of %d "
                           "windows that arm could reach, which are its "
                           "easiest. Its errors are **optimistically biased**, "
                           "so the gap above is understated."
                           % (label, arm["solved"], arm["windows"]))
    return (verdict, caveats)


def render(a, b):
    """The whole summary document."""
    lines = ["# E1 - structure versus no structure", ""]
    if a is None or b is None:
        lines.append("One or both arms produced no summary. Nothing to compare.")
        return "\n".join(lines) + "\n"

    def cell(value, fmt="%.2f", bold=False):
        if value is None:
            return "n/a"
        text = fmt % value
        return "**%s**" % text if bold else text

    lines += [
        "| | structured | unstructured |",
        "|---|---|---|",
        "| **windows solved** | **%d/%d** | **%d/%d** |"
        % (a["solved"], a["windows"], b["solved"], b["windows"]),
        "| windows with real motion | %d | %d |" % (a["scored"], b["scored"]),
        "| **median `mse_ratio`** (lower is better) | %s | %s |"
        % (cell(a["ratio"], bold=True), cell(b["ratio"], bold=True)),
        "| beats the straight line | %d | %d |" % (a["beats"], b["beats"]),
        "| trajectory IoU | %s | %s |"
        % (cell(a["iou"], "%.3f"), cell(b["iou"], "%.3f")),
        "| raw planner error *(not comparable across arms)* | %s | %s |"
        % (cell(a["mse"]), cell(b["mse"])),
        "| linear baseline *(shows why)* | %s | %s |"
        % (cell(a["base"]), cell(b["base"])),
        "",
    ]
    verdict, caveats = reading(a, b)
    lines.append(verdict)
    if caveats:
        lines.append("")
        for c in caveats:
            lines.append("- %s" % c)
    return "\n".join(lines) + "\n"


def main():
    a, b = read("structured"), read("unstructured")
    text = render(a, b)
    os.makedirs("eval/planner", exist_ok=True)
    out = "eval/planner/E1_summary.md"
    with open(out, "w") as handle:
        handle.write(text)
    print(text)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
