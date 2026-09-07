"""The one convention every oracle floor rests on: bins decode to left edges.

`oracle.dequantise` says in its own docstring that it maps a bin index the way
`common/decode.py` does, and records what it cost when the two disagreed: on
200 random two-object frames, 2026-08-30, the floor read **8.37 under bin
centres and 34.11 under the decoder's left edges**, a factor of 4.1. Every
statement of the form "the planner is at the floor" was measuring against a
floor no trained model could reach.

That was fixed. What was missing until 2026-09-06 was a test. The test that
claimed to guard it, `test_dequantise_matches_the_real_decoder`, asserted only
that `dequantise(idx, 60, 300) == idx * (300.0 / 60)` -- a hand-written
constant on one side and `dequantise` on the other. **No test anywhere
imported `common/decode.py`.** So `decode.py` could go back to bin centres and
the whole suite would stay green while the regression came back in silence.

This file runs both sides against each other. It fails if either one moves.

`decode.py` cannot be imported through the package, because that runs
`latplan/__init__.py` and pulls in TensorFlow. The geometry it needs is loaded
from the real `puzzle_labeled_objects.py` by file, which is what
`oracle.load_canvas_scaler` does, so the numbers here are the production
numbers and not a fixture.
"""

import importlib.util
import os
import sys
import types
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))
sys.path.insert(0, ROOT)

from tools.planner import box_geometry as oracle

_STUBBED = ("latplan", "latplan.util", "latplan.puzzles",
            "latplan.puzzles.puzzle_labeled_objects",
            "tools.planner.common.decode")


def _real_picsize():
    """`PICSIZE` from the real loader, read from file so TensorFlow stays out."""
    path = os.path.join(ROOT, "latplan", "puzzles", "puzzle_labeled_objects.py")
    spec = importlib.util.spec_from_file_location("_plo_for_decode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PICSIZE


class TestTheContract(unittest.TestCase):

    def setUp(self):
        self._saved = dict((k, sys.modules.get(k)) for k in _STUBBED)
        picsize = _real_picsize()

        for name in ("latplan", "latplan.util", "latplan.puzzles"):
            sys.modules.setdefault(name, types.ModuleType(name))
        plo = types.ModuleType("latplan.puzzles.puzzle_labeled_objects")
        plo.PICSIZE = picsize
        sys.modules["latplan.puzzles.puzzle_labeled_objects"] = plo

        path = os.path.join(ROOT, "tools", "planner", "common", "decode.py")
        spec = importlib.util.spec_from_file_location("_decode_direct", path)
        self.decode = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.decode)

        self.canvas_h, self.canvas_w = int(picsize[0]), int(picsize[1])

    def tearDown(self):
        for name, was in self._saved.items():
            if was is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = was

    # -- building a decoder output by hand ---------------------------------

    def _grid(self):
        grid = self.decode._picsize_grid()
        return int(grid[1]), int(grid[0])          # X, Y

    def _feat_dim(self, patch_len):
        x, y = self._grid()
        return patch_len + 2 * x + 2 * y

    def _features(self, bins, patch_len=12):
        """One frame, one object, with each coordinate's bin set to 1.

        `bins` is `(x1, y1, x2, y2)` as bin indices. Everything else is zero,
        which is what an argmax reads as bin 0.
        """
        x, y = self._grid()
        feat = np.zeros((1, 1, self._feat_dim(patch_len)), dtype=np.float32)
        offsets = [patch_len, patch_len + x, patch_len + x + y,
                   patch_len + 2 * x + y]
        for off, idx in zip(offsets, bins):
            feat[0, 0, off + idx] = 1.0
        return feat

    # -- the contract -------------------------------------------------------

    def test_the_decoder_and_the_oracle_agree_on_every_bin(self):
        """The assertion the suite was missing.

        Both sides are called. Neither is a constant written by hand. If
        `decode.py` returns to bin centres, or `oracle.dequantise` does, this
        goes red.
        """
        x, y = self._grid()
        for xi in range(0, x, 7):
            for yi in range(0, y, 5):
                got = self.decode.features_to_bboxes(
                    self._features((xi, yi, xi, yi)))[0, 0]
                self.assertAlmostEqual(
                    float(got[0]),
                    float(oracle.dequantise(xi, x, self.canvas_w)), places=4)
                self.assertAlmostEqual(
                    float(got[1]),
                    float(oracle.dequantise(yi, y, self.canvas_h)), places=4)

    def test_bin_zero_decodes_to_the_origin(self):
        """The left edge of bin 0 is 0. Its centre is not."""
        got = self.decode.features_to_bboxes(self._features((0, 0, 0, 0)))
        self.assertEqual(list(got[0, 0]), [0.0, 0.0, 0.0, 0.0])

    def test_it_is_the_left_edge_and_not_the_centre(self):
        """Named directly, because bin centre is what it used to be.

        Bin centre is the better estimator in isolation. It is not what the
        decoder emits, and the oracle exists to be the ceiling that trained
        models are measured against.
        """
        x, _ = self._grid()
        width = self.canvas_w / float(x)
        got = self.decode.features_to_bboxes(self._features((3, 0, 3, 0)))
        self.assertAlmostEqual(float(got[0, 0, 0]), 3 * width, places=4)
        self.assertNotAlmostEqual(float(got[0, 0, 0]), 3.5 * width, places=4)

    def test_the_last_bin_stops_short_of_the_canvas_edge(self):
        """A left-edge decoder can never emit the far edge, by construction."""
        x, _ = self._grid()
        got = self.decode.features_to_bboxes(
            self._features((x - 1, 0, x - 1, 0)))
        self.assertLess(float(got[0, 0, 0]), self.canvas_w)
        self.assertAlmostEqual(float(got[0, 0, 0]),
                               self.canvas_w - self.canvas_w / float(x),
                               places=4)

    def test_x_and_y_have_different_bin_counts(self):
        """The two axes are not interchangeable, though the bin width is.

        `PICSIZE // 5` makes a bin 5 pixels wide on both axes, so the scale
        factor alone cannot catch a transposition. The counts can: 60 against
        40. A bin index valid in x can be out of range in y, and the block
        offsets inside the feature vector depend on both.
        """
        x, y = self._grid()
        self.assertNotEqual(x, y)
        top = self.decode.features_to_bboxes(
            self._features((x - 1, y - 1, x - 1, y - 1)))[0, 0]
        self.assertAlmostEqual(
            float(top[0]), float(oracle.dequantise(x - 1, x, self.canvas_w)),
            places=4)
        self.assertAlmostEqual(
            float(top[1]), float(oracle.dequantise(y - 1, y, self.canvas_h)),
            places=4)
        self.assertNotAlmostEqual(float(top[0]), float(top[1]), places=4)

    def test_the_bbox_block_sits_after_the_patch_whatever_its_length(self):
        """The layout claim, at two patch sizes.

        `_feat_layout` derives the patch length by subtraction, so a wrong
        bbox length would shift every slice and quietly mis-read the boxes.
        """
        for patch_len in (0, 12, 3072):
            got = self.decode.features_to_bboxes(
                self._features((5, 4, 9, 7), patch_len=patch_len))[0, 0]
            x, y = self._grid()
            self.assertAlmostEqual(
                float(got[0]), float(oracle.dequantise(5, x, self.canvas_w)),
                places=4)
            self.assertAlmostEqual(
                float(got[3]), float(oracle.dequantise(7, y, self.canvas_h)),
                places=4)


if __name__ == "__main__":
    unittest.main()
