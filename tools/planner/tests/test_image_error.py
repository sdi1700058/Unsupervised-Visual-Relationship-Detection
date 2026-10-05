"""The reconstruction error of the image part and of the box part, apart.

A FOSAE video state holds, for each object, its image patch and then a
one-hot code of its box. The box code is 51 per cent of the loss at patch 8
and 3 per cent at patch 48 (`git show 56dac44:experiments/G5_patch_size/
README.md`), so a whole-vector MSE is not comparable across patch sizes. The
image part alone is. Added 2026-10-05, plan
`notes/plans/2026-10-04-overfit-viability.md`, Task 5.
"""

import unittest

import numpy as np

from tools import image_error as ie


class SplitErrors(unittest.TestCase):

    def test_the_image_and_box_parts_are_scored_apart(self):
        x = np.zeros((2, 3, 6), dtype=np.float32)
        y = x.copy()
        y[..., :4] = 0.5      # every image value off by 0.5
        got = ie.split_errors(x, y, image_dim=4)
        self.assertAlmostEqual(got["image_MSE"], 0.25)
        self.assertAlmostEqual(got["box_MSE"], 0.0)
        self.assertAlmostEqual(got["all_MSE"], 0.25 * 4 / 6)

    def test_a_perfect_copy_has_no_error(self):
        x = np.random.RandomState(0).rand(4, 3, 10).astype(np.float32)
        got = ie.split_errors(x, x.copy(), image_dim=7)
        self.assertEqual(got, {"image_MSE": 0.0, "box_MSE": 0.0,
                               "all_MSE": 0.0})


if __name__ == "__main__":
    unittest.main()
