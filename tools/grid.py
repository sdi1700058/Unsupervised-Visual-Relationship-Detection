#!/usr/bin/env python3
"""The evaluation grid: which dataset by method by source has been measured.

**One combination is one dataset, by one evaluation method, by one source.**
`oracle` means the latents encode ground-truth boxes, so it answers what a
planner would output given ideal relations. `trained` means a network produced
them. The two answer different questions, which is why the supervisor asked for
the problem to be split that way.

**Four empty states, and they never merge.**

    never run     nothing has been attempted here yet
    cannot run    the cell cannot exist, and the reason is given
    collapsed     it ran, and the encoder emitted one code for every frame
    scored        it ran and produced evidence that is on disk

Merging any two of these is how a dead export became a planning result. A cell
that cannot exist is not a cell that scored badly, and a collapsed run is an
absent measurement rather than a poor one.

**A trained cell needs frames; an oracle cell does not.** The oracle reads
boxes, which is the fact that makes a second and a third dataset cost days
rather than weeks. Measured on 2026-09-05, only VidVRD had extracted frames, so
most trained cells report `cannot run` rather than failing when attempted.

**The default is a dry run.** It prints the cells and the commands and submits
nothing, because a queue filled without the author seeing it is a queue they
cannot plan around.

    python3 tools/grid.py
    python3 tools/grid.py --source oracle

Standard library only, Python 3.6 clean.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEVER_RUN = "never run"
CANNOT_RUN = "cannot run"
COLLAPSED = "collapsed"
SCORED = "scored"

STATES = (SCORED, COLLAPSED, NEVER_RUN, CANNOT_RUN)

SOURCES = ("oracle", "trained")


def _evidence_index(plan, root="."):
    """`{(dataset, method, source): evidence path}` for files that exist."""
    index = {}
    for combo in plan.get("combinations", []):
        evidence = combo.get("evidence", "")
        if not evidence:
            continue
        if not os.path.exists(os.path.join(root, evidence)):
            continue
        key = (combo["dataset"], combo["method"],
               combo.get("source", "oracle"))
        index[key] = evidence
    return index


def _is_collapsed(export, root="."):
    """Whether a named export holds one distinct latent.

    Imported lazily. numpy is not needed to enumerate a grid, and a machine
    without it should still be able to see what has been measured.
    """
    if not export:
        return False
    path = os.path.join(root, export)
    if not os.path.exists(path):
        return False
    try:
        from tools.planner import liveness
    except ImportError:
        return False
    return bool(liveness.export_facts(path).get("dead"))


def enumerate_cells(plan, root="."):
    """One row per dataset, method and source, with its state and the reason."""
    datasets = plan.get("datasets_for_grid") or []
    methods = plan.get("methods_for_grid") or []
    evidence = _evidence_index(plan, root)

    cells = []
    for dataset in datasets:
        name = dataset.get("name")
        has_frames = bool(dataset.get("frames"))
        for method in methods:
            for source in SOURCES:
                key = (name, method, source)
                found = evidence.get(key)
                if found:
                    state, reason = SCORED, found
                elif source == "trained" and not has_frames:
                    # Training needs pixels. Saying so is different from
                    # trying and failing, and the difference is the whole
                    # point of keeping four states.
                    state = CANNOT_RUN
                    reason = ("no extracted frames for %s, and training needs "
                              "them; the oracle does not" % name)
                elif _is_collapsed(dataset.get("export"), root):
                    state = COLLAPSED
                    reason = ("the export holds one distinct latent, so there "
                              "is no state to plan over")
                else:
                    state, reason = NEVER_RUN, ""
                cells.append({"dataset": name, "method": method,
                              "source": source, "state": state,
                              "reason": reason})
    return cells


def missing(cells):
    """The cells that could run and have not."""
    return [c for c in cells if c["state"] == NEVER_RUN]


def summarise(cells):
    """`{state: count}` over every state, including the ones at zero."""
    counts = dict((state, 0) for state in STATES)
    for cell in cells:
        counts[cell["state"]] = counts.get(cell["state"], 0) + 1
    return counts


# How each method is actually run, as a template.
#
# **These are the real command lines, not a convention.** The first version
# emitted `--dataset X --source Y` for every method, and no runner accepts
# those flags: `m4_temporal.py` takes positional paths, and the others differ
# again. A command that cannot run is worse than no command, because it looks
# like work waiting to happen.
#
# `{export}` is the dataset's oracle latents, `{ann}` its annotation root, and
# `{out}` the output directory. A template naming a placeholder the dataset
# cannot fill produces no command, and the reason is reported.
RUNNERS = {
    "temporal-distance":
        "{py} tools/planner/m4_temporal.py {name}-{source}={export} "
        "--out-dir {out}",
    # Pooled, not one row per clip. m6 takes LABEL=a,b,c to treat several
    # exports as one dataset; passing a directory reads nothing and passing
    # each file separately would measure 25 datasets of one clip each.
    "determinism":
        "{py} tools/planner/m6_determinism.py "
        "\"{name}-{source}=$(ls {export}/*.npz | paste -sd,)\" "
        "--out-dir {out}",
    "triplet-mAP":
        "{py} tools/planner/m7_map.py --annotations {ann} --export {export} "
        "--out-dir {out}",
    "probing":
        "{py} tools/planner/predicate_probe.py --annotation {ann} "
        "--out-dir {out}",
    "compositional":
        "{py} tools/planner/compositional.py --export {export} "
        "--out-dir {out}",
    "plan-validity":
        "{py} tools/planner/plan_validity.py {export}/*.npz --out-dir {out}",
    "interpolation":
        "bash tools/planner/eval_plannability.sh {export} --window 16",
}

# The interpreter that has pillow. Bare python3 silently skips work that needs
# it, and a silent skip has already been read as a pass on this project.
PY = ".venv-local/bin/python"


def plan_commands(cells, datasets=None):
    """One command per runnable cell, and nothing for a cell that cannot run."""
    where = {}
    for dataset in datasets or []:
        where[dataset.get("name")] = dataset

    commands = []
    for cell in cells:
        if cell["state"] != NEVER_RUN:
            continue
        template = RUNNERS.get(cell["method"])
        if template is None:
            commands.append(
                "# no runner yet for %s (%s, %s). Write one before this cell "
                "can run." % (cell["method"], cell["dataset"], cell["source"]))
            continue
        dataset = where.get(cell["dataset"], {})
        export = dataset.get("oracle_export") or ""
        ann = dataset.get("annotations") or ""
        if "{export}" in template and not export:
            commands.append(
                "# %s has no oracle export yet, so %s cannot run on it. Build "
                "one with tools/planner/oracle.py."
                % (cell["dataset"], cell["method"]))
            continue
        if "{ann}" in template and not ann:
            commands.append(
                "# %s has no annotation root recorded, so %s cannot run."
                % (cell["dataset"], cell["method"]))
            continue
        out = "eval/grid/%s-%s-%s" % (cell["dataset"],
                                      cell["method"].split()[0],
                                      cell["source"])
        commands.append(template.format(py=PY, name=cell["dataset"],
                                        source=cell["source"], export=export,
                                        ann=ann, out=out))
    return commands


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", default="notes/WORKPLAN.json")
    ap.add_argument("--source", default=None, choices=list(SOURCES),
                    help="show one source only")
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.plan):
        print("no plan at %s" % a.plan)
        return 2
    with open(a.plan) as handle:
        plan = json.load(handle)

    cells = enumerate_cells(plan, root=a.root)
    if a.source:
        cells = [c for c in cells if c["source"] == a.source]
    if not cells:
        print("no cells. Add datasets_for_grid and methods_for_grid to the "
              "plan.")
        return 0

    counts = summarise(cells)
    print("%d cell(s): %s\n"
          % (len(cells),
             ", ".join("%d %s" % (counts[s], s) for s in STATES)))

    width = max(len(c["dataset"]) for c in cells)
    for cell in cells:
        line = ("  %-*s  %-22s %-8s %s"
                % (width, cell["dataset"], cell["method"], cell["source"],
                   cell["state"]))
        print(line)
        if cell["reason"] and cell["state"] != SCORED:
            print("  %-*s    %s" % (width, "", cell["reason"]))

    gaps = missing(cells)
    if gaps:
        print("\n%d cell(s) could run now. Nothing is submitted; this is a "
              "dry run.\n" % len(gaps))
        for command in plan_commands(
                gaps, plan.get('datasets_for_grid')):
            print("    %s" % command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
