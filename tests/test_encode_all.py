"""Encoding every state, in batches that fit in memory.

`export_latents.py` could not run on a real clip set: `encode_all` passed
all 14,314 states of the KIT role clips to `ae.encode` in one call and the
job was killed at 8 GB on 2026-10-10. Batching changes no number, so the
tests pin that the batched answer equals the whole-array answer.
"""

import unittest

import numpy as np

from planner.common.encode import apply_in_batches, encode_all


class _Recorder(object):
    """An autoencoder that rounds, and remembers each call's size."""

    def __init__(self, fail_above=None):
        self.sizes = []
        self.fail_above = fail_above

    def encode(self, states):
        if self.fail_above is not None and len(states) > self.fail_above:
            raise MemoryError("%d states at once" % len(states))
        self.sizes.append(len(states))
        return np.asarray(states, dtype=float) * 0.6


class EncodeAll(unittest.TestCase):

    def test_the_batched_answer_equals_the_whole_array_answer(self):
        states = np.arange(20, dtype=float).reshape(10, 2)
        whole = encode_all(_Recorder(), states)
        batched = encode_all(_Recorder(), states, batch=3)
        np.testing.assert_array_equal(whole, batched)

    def test_it_encodes_in_batches_of_the_size_asked_for(self):
        ae = _Recorder()
        encode_all(ae, np.zeros((10, 2)), batch=4)
        self.assertEqual(ae.sizes, [4, 4, 2])

    def test_a_model_that_cannot_take_the_whole_array_still_finishes(self):
        ae = _Recorder(fail_above=4)
        out = encode_all(ae, np.ones((10, 2)), batch=4)
        self.assertEqual(out.shape, (10, 2))

    def test_the_result_is_rounded_to_int8(self):
        out = encode_all(_Recorder(), np.ones((4, 2)), batch=2)
        self.assertEqual(out.dtype, np.int8)
        # 1.0 * 0.6 rounds to 1
        self.assertTrue((out == 1).all())

    def test_no_batch_means_one_call(self):
        ae = _Recorder()
        encode_all(ae, np.zeros((10, 2)))
        self.assertEqual(ae.sizes, [10])

    def test_an_empty_state_array_gives_an_empty_result(self):
        self.assertEqual(encode_all(_Recorder(), np.zeros((0, 2)),
                                    batch=4).shape, (0, 2))


class ApplyInBatches(unittest.TestCase):
    """The same batching for `autoencode`, which `export_latents.py` also
    called on the whole array."""

    def test_the_batched_answer_equals_the_whole_array_answer(self):
        states = np.arange(24, dtype=float).reshape(12, 2)
        ae = _Recorder()
        np.testing.assert_array_equal(
            apply_in_batches(ae.encode, states, None),
            apply_in_batches(ae.encode, states, 5))

    def test_it_keeps_the_float_values(self):
        out = apply_in_batches(_Recorder().encode, np.ones((4, 2)), 2)
        self.assertEqual(out.dtype, float)
        self.assertTrue((out == 0.6).all())

    def test_an_empty_array_gives_an_empty_result(self):
        self.assertEqual(
            apply_in_batches(_Recorder().encode, np.zeros((0, 3)), 2).shape,
            (0, 3))


if __name__ == "__main__":
    unittest.main()
