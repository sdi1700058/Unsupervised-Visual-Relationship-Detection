"""Keep the weights of the epoch with the lowest val value.

Found 2026-10-06: the runs at patch 48 reached their lowest val BCE near
epoch 1,000 of 5,000 and overfit after it, while every run saved only the
weights of its last epoch, so the best weights could not be tested.
`latplan.model` updates a keeper each epoch when `KEEP_BEST=1` and restores
the kept weights before it saves. The weights are copied in memory, so no
file is written each epoch.

This module imports nothing from `latplan`, so it loads without TensorFlow.
"""

import math

import numpy as np


class BestKeeper(object):
    """The epoch, the value and a copy of the weights of the lowest value."""

    def __init__(self, start=0):
        # The val BCE is high and unstable until the pre-encoder fade-in
        # ends and spikes at its end, so `latplan.model` sets `start` there.
        self.start = int(start)
        self.epoch = None
        self.value = None
        self.weights = None

    def update(self, epoch, value, get_weights):
        """Keep `get_weights()` when `value` is the lowest so far."""
        value = float(value)
        if epoch < self.start or math.isnan(value):
            return False
        if self.value is not None and value >= self.value:
            return False
        self.epoch, self.value = int(epoch), value
        self.weights = [np.array(w, copy=True) for w in get_weights()]
        return True


def split_by_counts(flat, counts):
    """`flat`, the weights of several networks in a row, back into one list
    for each network, `counts[i]` weights for network `i`."""
    if sum(counts) != len(flat):
        raise ValueError("%d weights for counts that add up to %d"
                         % (len(flat), sum(counts)))
    out, k = [], 0
    for n in counts:
        out.append(list(flat[k:k + n]))
        k += n
    return out
