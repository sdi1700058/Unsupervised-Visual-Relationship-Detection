#!/usr/bin/env python3
"""Read a planner export and print each state's relations over objects.

THIS IS THE COMMAND THAT MAKES STAGE ONE COMPUTABLE. The supervisors set a
two-stage evaluation on 2026-09-15 and asked, for the relation stage, for
*"a list of predicates with objects as arguments"* over a given object set O.

Until 2026-09-21 `export_latents.py` wrote no `bindings` key, so a FOSAE bit
inverted to (unit, predicate) and stopped, while `relation_oracle.py` wrote
`predicates` and `num_objs` and could name the objects. One arm had the shape
their record asks for and the other did not, so the two could not be compared
on any dataset. That is why this is dataset-independent work and why it runs
before the dataset choice.

    python3 tools/planner/show_relations.py <planner_export.npz>
    python3 tools/planner/show_relations.py <export> --frame <id> --confidence

It needs numpy and nothing else, so it runs wherever the export does,
including Sherlock's Python 3.6. It loads with `allow_pickle=False`.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))

from tools.planner.export_latents import invert_bit  # noqa: E402

MISSING = (
    "this export has no `bindings` key, so a bit inverts to (unit, "
    "predicate) and stops. Every export written before 2026-09-21 lacks it. "
    "Re-run tools/planner/export_latents.py to produce one that carries it.")


def render(path, frame=None, with_confidence=False):
    """The report, as a string, so a test reads it without a subprocess."""
    data = np.load(path, allow_pickle=False)
    if "bindings" not in data:
        raise KeyError(MISSING)

    latents = data["latents"]
    bindings = data["bindings"]
    P = int(data["P"])
    confidence = data["binding_confidence"] if with_confidence else None
    ids = ([str(f) for f in data["frame_ids"]] if "frame_ids" in data
           else [str(i) for i in range(len(latents))])

    lines = []
    shown = 0
    for i, name in enumerate(ids):
        if frame is not None and name != frame:
            continue
        shown += 1
        truth = np.round(latents[i]).astype(int)
        found = [(bit, invert_bit(bit, bindings[i], P))
                 for bit in range(len(truth)) if truth[bit]]
        if not found:
            # An empty answer and a silent one must not look the same.
            lines.append("%s: no relation is true in this state" % name)
            continue
        lines.append("%s: %d relation(s)" % (name, len(found)))
        for bit, tup in found:
            row = "    bit %-4d -> %s" % (bit, tup)
            if confidence is not None:
                unit = bit // P
                mass = ", ".join("%.2f" % c for c in confidence[i][unit])
                row += "   attention %s" % mass
            lines.append(row)

    if frame is not None and shown == 0:
        lines.append("no state is named %r in this export" % frame)
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("export", help="a planner_export.npz")
    ap.add_argument("--frame", default=None, help="one frame id")
    ap.add_argument("--confidence", action="store_true",
                    help="print the attention mass behind each binding")
    args = ap.parse_args(argv)
    try:
        sys.stdout.write(render(args.export, args.frame, args.confidence))
    except KeyError as problem:
        sys.stderr.write("show_relations: %s\n" % problem.args[0])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
