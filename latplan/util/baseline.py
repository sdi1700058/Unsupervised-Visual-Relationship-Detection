"""The loss of a model that always predicts the mean training state.

A reconstruction curve read alone says nothing: a val BCE of 0.12 can be good
or trivial. This is the trivial model, scored the way the network scores
itself (`latplan/util/distances.py`: Keras `binary_crossentropy` over the
flattened state, prediction clipped to [1e-7, 1 - 1e-7], averaged over
features and then over samples). A val curve under `val_BCE` here learned
more than the average frame. Added 2026-10-03.

This module imports nothing from `latplan`, so it loads without TensorFlow.
"""

import numpy as np

EPSILON = 1e-7


def _scores(mean, states):
    x = states.reshape(len(states), -1).astype(np.float64)
    p = np.clip(mean, EPSILON, 1.0 - EPSILON)
    bce = -(x * np.log(p) + (1.0 - x) * np.log(1.0 - p))
    return float(bce.mean()), float(np.square(x - mean).mean())


def mean_baseline(train, val):
    """`{train_BCE, train_MSE, val_BCE, val_MSE, rule}` for the mean predictor."""
    mean = train.reshape(len(train), -1).astype(np.float64).mean(axis=0)
    train_bce, train_mse = _scores(mean, train)
    val_bce, val_mse = _scores(mean, val)
    return {"train_BCE": train_bce, "train_MSE": train_mse,
            "val_BCE": val_bce, "val_MSE": val_mse,
            "rule": "always predict the mean training state, scored as the "
                    "network's BCE and MSE metrics"}
