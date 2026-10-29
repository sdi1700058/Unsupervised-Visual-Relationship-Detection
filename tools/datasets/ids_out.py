#!/usr/bin/env python3
"""One place to open an `--ids-out` list, so a rerun cannot silently lose one.

`notes/OPEN-ISSUES.md` I28 (archived, see workbench/ARCHIVE.md). Two tracked
ids files went this way in six days,
`scripts/train/rung_ids/kit3d-roles.txt` on 2026-10-06 and
`scripts/train/rung_ids/se-250clips.txt` on 2026-10-11. Both times a rerun pointed
`--ids-out` at a list somebody else had made, both times the run succeeded,
and both times the damage was found later by eye. Four converters each held
their own `open(args.ids_out, "w")`, which is why there was nowhere to put
the check.

A converter calls `open_ids_out(path, force)` instead of `open(path, "w")`
and adds `--force-ids-out` beside its `--ids-out`.
"""

import os


def open_ids_out(path, force=False):
    """Open `path` for writing, refusing to overwrite a file already there.

    Refuses with SystemExit BEFORE opening, so the file on disk is untouched.
    Opening with "w" first and checking afterwards would truncate the very
    thing this protects.
    """
    if os.path.exists(path) and not force:
        raise SystemExit(
            "refusing to overwrite the ids list at %s\n"
            "Something already wrote that file. Two tracked lists were lost "
            "this way in six days, notes/OPEN-ISSUES.md I28 "
            "(archived, see workbench/ARCHIVE.md).\n"
            "Write somewhere else, or pass --force-ids-out to overwrite it "
            "on purpose." % path)
    return open(path, "w")
