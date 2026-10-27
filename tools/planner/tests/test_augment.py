"""`_clip_boxes` promised "non-degenerate" boxes and did not keep it.

A box translated off-canvas clips both edges to the same column: x1 == x2
survives, zero width, but is not all-zero so `_padding_mask` still counts it
as a real slot. KEEP_FIX A#2: a degenerate box after clipping becomes a
zeroed, padding slot.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, os.pardir, os.pardir))

from latplan.util.augment import translate, _padding_mask


class TestDegenerateBoxesBecomePadding(unittest.TestCase):

    def test_a_box_pushed_fully_off_canvas_is_zeroed_and_padded(self):
        width, height = 20, 20
        # One real box, far right, that a big rightward shift pushes past
        # the canvas edge on both corners.
        boxes = np.array([[[5.0, 5.0, 8.0, 8.0]]], dtype=np.float32)
        images = np.zeros((1, height, width, 3), dtype=np.float32)
        _, out_boxes = translate(images, boxes, width, height, dx=1000, dy=0)
        self.assertTrue(np.all(out_boxes[0, 0] == 0),
                        "degenerate box was not zeroed: %r" % (out_boxes[0, 0],))
        self.assertFalse(_padding_mask(out_boxes)[0, 0],
                         "degenerate box still reads as a real slot")


if __name__ == "__main__":
    unittest.main()
