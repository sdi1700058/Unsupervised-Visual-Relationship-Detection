#!/usr/bin/env python3
"""A clip list must record how it was made. SPEC V37.

`eval/vidvrd_winnable_clips.txt` is 88 bare ids with no header. Nothing in it
says which window screened it, and the current screen reproduces no window
exactly -- w=12 gives 69, w=16 gives 86, w=20 gives 99, and the closest shares
80 of the 88. The provenance is unrecoverable, and every planner number in the
project was scored at a window the list may never have been screened at.

That is not a bug in the screen; it is a missing field.

    python3 -m unittest tools/planner/tests/test_list_provenance.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

_VIDEO = str(Path(__file__).resolve().parents[2] / "video")
if _VIDEO not in sys.path:
    sys.path.insert(0, _VIDEO)


class TestProvenanceHeader(unittest.TestCase):

    def test_the_header_records_the_window(self):
        from screen_vidvrd import provenance_header

        h = provenance_header(window=16, n=88, filters={"no_fill_only": True,
                                                        "min_frames": 45})
        self.assertTrue(h.startswith("#"))
        self.assertIn("window=16", h)
        self.assertIn("88", h)

    def test_every_line_is_a_comment(self):
        """So the file still reads as a plain id list."""
        from screen_vidvrd import provenance_header

        for line in provenance_header(window=8, n=39, filters={}).splitlines():
            self.assertTrue(line.startswith("#") or not line.strip())

    def test_the_filters_appear(self):
        from screen_vidvrd import provenance_header

        h = provenance_header(window=8, n=39,
                              filters={"min_frames": 45, "no_fill_only": True})
        self.assertIn("min_frames=45", h)
        self.assertIn("no_fill_only=True", h)

    def test_a_reader_skips_comments(self):
        from screen_vidvrd import read_clip_list

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "list.txt")
            with open(p, "w") as f:
                f.write("# screened at window=16\n#\nCLIP_A\n\nCLIP_B\n")
            self.assertEqual(read_clip_list(p), ["CLIP_A", "CLIP_B"])

    def test_the_reader_returns_the_window_it_finds(self):
        """So a caller can refuse to score at the wrong one."""
        from screen_vidvrd import clip_list_window

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "list.txt")
            with open(p, "w") as f:
                f.write("# screen_vidvrd.py window=16 n=88\nCLIP_A\n")
            self.assertEqual(clip_list_window(p), 16)

    def test_a_list_with_no_header_reports_unknown(self):
        """Which is the state eval/vidvrd_winnable_clips.txt is in."""
        from screen_vidvrd import clip_list_window

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "list.txt")
            with open(p, "w") as f:
                f.write("CLIP_A\nCLIP_B\n")
            self.assertIsNone(clip_list_window(p))


if __name__ == "__main__":
    unittest.main()
