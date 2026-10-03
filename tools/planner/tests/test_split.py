"""Train, val and test hold whole clips, or val measures nothing new.

`strips.py vidvrd()` built its split from `transitions.reshape(...)`, which
puts every "before" state first and every "after" state second, then cut that
array 90/5/5. Val therefore held the "after" frames of the last clips, and
frame t+1 of one pair is frame t of the next pair in the same clip, so nearly
every val state was also a training state. Found 2026-10-03 while planning
the rung runs, and confirmed by an adversary from `strips.py:622-628`.

`SPLIT=clip` gives each clip to exactly one split. The default is unchanged,
so earlier runs reproduce. `latplan/util/split.py` imports nothing from
`latplan`, so it loads from file here without TensorFlow.
"""

import importlib.util
import os
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir, os.pardir, os.pardir))

_spec = importlib.util.spec_from_file_location(
    "_split", os.path.join(ROOT, "latplan", "util", "split.py"))
split = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split)


def _clips(n_clips, frames_each):
    """Frame ids and the consecutive pairs of `n_clips` clips, in load order."""
    frame_ids, pairs = [], []
    for c in range(n_clips):
        start = len(frame_ids)
        for f in range(frames_each):
            frame_ids.append("clip%03d/%06d" % (c, f))
        pairs += [(i, i + 1) for i in range(start, start + frames_each - 1)]
    return frame_ids, pairs


class TestAssign(unittest.TestCase):

    def test_twenty_clips_give_eighteen_one_and_one_in_load_order(self):
        frame_ids, _ = _clips(20, 3)
        got = split.assign_clips(frame_ids)
        self.assertEqual([c for c, s in got.items() if s == "val"], ["clip018"])
        self.assertEqual([c for c, s in got.items() if s == "test"], ["clip019"])
        self.assertEqual(sum(1 for s in got.values() if s == "train"), 18)

    def test_a_hundred_clips_give_ninety_five_and_five(self):
        frame_ids, _ = _clips(100, 2)
        got = split.assign_clips(frame_ids)
        counts = {s: sum(1 for v in got.values() if v == s)
                  for s in ("train", "val", "test")}
        self.assertEqual(counts, {"train": 90, "val": 5, "test": 5})

    def test_fewer_than_three_clips_cannot_be_split(self):
        frame_ids, _ = _clips(2, 5)
        with self.assertRaises(ValueError):
            split.assign_clips(frame_ids)


class TestSplitStates(unittest.TestCase):

    def test_no_val_or_test_frame_reaches_training(self):
        frame_ids, pairs = _clips(20, 4)
        states = np.arange(len(frame_ids), dtype=np.float32).reshape(-1, 1, 1)
        train, val, test, info = split.split_states(states, frame_ids, pairs)
        held = set(np.concatenate([val, test]).ravel().tolist())
        self.assertFalse(held & set(train.ravel().tolist()))

    def test_each_split_holds_the_before_then_the_after_state_of_its_pairs(self):
        frame_ids, pairs = _clips(20, 4)
        states = np.arange(len(frame_ids), dtype=np.float32).reshape(-1, 1, 1)
        _train, val, _test, info = split.split_states(states, frame_ids, pairs)
        # clip018 holds frames 72..75, so pairs (72,73) (73,74) (74,75)
        self.assertEqual(val.ravel().tolist(), [72, 73, 74, 73, 74, 75])
        self.assertEqual(info["pairs"], {"train": 54, "val": 3, "test": 3})
        self.assertEqual(info["clips"], {"train": 18, "val": 1, "test": 1})

    def test_a_split_with_no_pair_is_refused_by_name(self):
        frame_ids, pairs = _clips(20, 4)
        # the last clip is reduced to one frame, so the test split has no pair
        frame_ids = frame_ids[:-3]
        pairs = [p for p in pairs if p[1] < len(frame_ids)]
        states = np.zeros((len(frame_ids), 1, 1), dtype=np.float32)
        with self.assertRaises(ValueError) as caught:
            split.split_states(states, frame_ids, pairs)
        self.assertIn("test", str(caught.exception))


class TestFramesOf(unittest.TestCase):
    """`tools/replot.py` redraws `render_test.png` after training. It sampled
    any 200 states of the npz, so most of its "test" pictures were training
    frames. With the clip split recorded in the run manifest it now takes its
    test states from the test clips only, through `frames_of`."""

    def test_the_indices_of_the_named_clips_come_back_in_order(self):
        frame_ids, _ = _clips(4, 2)
        self.assertEqual(split.frames_of(frame_ids, ["clip001", "clip003"]),
                         [2, 3, 6, 7])

    def test_no_named_clip_gives_no_index(self):
        frame_ids, _ = _clips(3, 2)
        self.assertEqual(split.frames_of(frame_ids, []), [])


if __name__ == "__main__":
    unittest.main()
