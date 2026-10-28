#!/usr/bin/env python3
"""A converter must not silently overwrite an ids list it did not write.

WHY THIS EXISTS. `notes/OPEN-ISSUES.md` I28. Two TRACKED ids files were lost
this way inside six days, each time by a rerun whose `--ids-out` pointed at a
list somebody else had made:

  2026-10-06  scripts/train/rung_ids/kit3d-roles.txt   restored by the author, by hand
  2026-10-11  scripts/train/rung_ids/se-250clips.txt   238 wrong ids sat in the working
                                            tree until a session noticed

Neither run failed. Both wrote a perfectly good file over a perfectly good
file, and the only reason nothing broke is that nothing happened to read the
clobbered list first. Four converters each carried their own
`open(args.ids_out, "w")`, so there was no single place to put the check.
"""

import os
import shutil
import tempfile
import unittest

from tools.video import ids_out


class Guard(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.path = os.path.join(self.dir, "ids.txt")

    def test_a_new_path_opens_for_writing(self):
        with ids_out.open_ids_out(self.path) as handle:
            handle.write("clip_a\n")
        self.assertEqual(open(self.path).read(), "clip_a\n")

    def test_an_existing_path_is_refused(self):
        open(self.path, "w").write("clip_a\n")
        with self.assertRaises(SystemExit):
            ids_out.open_ids_out(self.path)

    def test_the_refusal_leaves_the_file_untouched(self):
        # The whole point. A refusal that truncated first would be the bug.
        open(self.path, "w").write("clip_a\nclip_b\n")
        try:
            ids_out.open_ids_out(self.path)
        except SystemExit:
            pass
        self.assertEqual(open(self.path).read(), "clip_a\nclip_b\n")

    def test_the_refusal_names_the_path_and_the_way_round_it(self):
        open(self.path, "w").write("clip_a\n")
        with self.assertRaises(SystemExit) as caught:
            ids_out.open_ids_out(self.path)
        message = str(caught.exception)
        self.assertIn(self.path, message)
        self.assertIn("--force-ids-out", message)

    def test_force_overwrites(self):
        open(self.path, "w").write("clip_a\n")
        with ids_out.open_ids_out(self.path, force=True) as handle:
            handle.write("clip_b\n")
        self.assertEqual(open(self.path).read(), "clip_b\n")

    def test_an_empty_existing_file_is_still_refused(self):
        # Zero bytes is still somebody's file, and a half-written list is
        # exactly the state worth protecting.
        open(self.path, "w").close()
        with self.assertRaises(SystemExit):
            ids_out.open_ids_out(self.path)


if __name__ == "__main__":
    unittest.main()
