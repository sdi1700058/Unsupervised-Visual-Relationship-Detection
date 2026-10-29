"""Train, val and test hold whole clips, or val measures nothing new.

`model/train/strips.py vidvrd()` built its split from `transitions.reshape(...)`, which
puts every "before" state first and every "after" state second, then cut that
array 90/5/5. Val therefore held the "after" frames of the last clips, and
frame t+1 of one pair is frame t of the next pair in the same clip, so nearly
every val state was also a training state. Found 2026-10-03 while planning
the rung runs, and confirmed by an adversary from `model/train/strips.py:622-628`.

`SPLIT=clip` gives each clip to exactly one split. The default is unchanged,
so earlier runs reproduce. `model/loaders/split.py` imports nothing from
the `model` package, so it loads from file here without TensorFlow.
"""

import importlib.util
import os
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    os.pardir))

_spec = importlib.util.spec_from_file_location(
    "_split", os.path.join(ROOT, "model", "loaders", "split.py"))
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


def _session_clips(sessions):
    """Frame ids of clips named like HA-ViD windows, `S25A05I01M0_000150`."""
    frame_ids = []
    for session, n_clips in sessions:
        for c in range(n_clips):
            for f in range(2):
                frame_ids.append("%sA%02dI01M0_%06d/%06d" % (session, c, 0, f))
    return frame_ids


class TestAssignByGroup(unittest.TestCase):
    """SPLIT_GROUP keeps one session in one split. Added 2026-10-05.

    HA-ViD windows load in sorted name order, so the 4 val windows of the
    2026-10-03 runs came from session S25, which training also held. A
    regex on the clip name names its group, and whole groups go to val and
    test, so a val clip shows a session that training never saw.
    """

    def test_a_session_never_spans_two_splits(self):
        frame_ids = _session_clips([("S04", 6), ("S13", 6), ("S25", 4),
                                    ("S28", 4)])
        got = split.assign_clips(frame_ids, group="S[0-9]+")
        by_session = {}
        for clip, name in got.items():
            by_session.setdefault(clip[:3], set()).add(name)
        self.assertEqual(by_session, {"S04": {"train"}, "S13": {"train"},
                                      "S25": {"val"}, "S28": {"test"}})

    def test_a_large_last_session_goes_whole_to_val(self):
        frame_ids = _session_clips([("S04", 30), ("S25", 12), ("S28", 4)])
        got = split.assign_clips(frame_ids, group="S[0-9]+")
        val = [c for c, s in got.items() if s == "val"]
        self.assertEqual(len(val), 12)
        self.assertTrue(all(c.startswith("S25") for c in val))

    def test_a_name_the_regex_misses_is_its_own_group(self):
        frame_ids, _ = _clips(20, 2)
        got = split.assign_clips(frame_ids, group="S[0-9]+")
        self.assertEqual(got, split.assign_clips(frame_ids))

    def test_named_groups_go_to_val_and_test(self):
        frame_ids = _session_clips([("S04", 6), ("S21", 2), ("S25", 12),
                                    ("S28", 2)])
        got = split.assign_clips(frame_ids, group="S[0-9]+",
                                 val_groups=["S21"], test_groups=["S28"])
        by_session = {}
        for clip, name in got.items():
            by_session.setdefault(clip[:3], set()).add(name)
        self.assertEqual(by_session, {"S04": {"train"}, "S21": {"val"},
                                      "S25": {"train"}, "S28": {"test"}})

    def test_a_named_group_missing_from_the_load_is_refused(self):
        frame_ids = _session_clips([("S04", 6), ("S21", 2), ("S28", 2)])
        with self.assertRaises(ValueError):
            split.assign_clips(frame_ids, group="S[0-9]+",
                               val_groups=["S99"], test_groups=["S28"])

    def test_fewer_than_three_groups_cannot_be_split(self):
        frame_ids = _session_clips([("S04", 6), ("S25", 6)])
        with self.assertRaises(ValueError):
            split.assign_clips(frame_ids, group="S[0-9]+")


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
    """`tools/analysis/replot.py` redraws `render_test.png` after training. It sampled
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
