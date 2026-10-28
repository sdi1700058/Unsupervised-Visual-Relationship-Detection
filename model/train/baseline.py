"""The loss of a model that always predicts the mean training state.

A reconstruction curve read alone says nothing: a val BCE of 0.12 can be good
or trivial. This is the trivial model, scored the way the network scores
itself (`model/losses.py`: Keras `binary_crossentropy` over the
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


def bce_floor(states):
    """The mean binary entropy of the targets: the BCE of a perfect copy.

    BCE reaches 0 only on targets of exactly 0 or 1. Video patches are
    equalised and stretched into values between 0 and 1, so a perfect copy
    of a video state still scores this. Clipped as the network clips its
    prediction, so a target of exactly 0 or 1 adds about 1e-07 and not 0.
    Added 2026-10-05, plan notes/plans/2026-10-04-overfit-viability.md.
    """
    x = np.clip(states.reshape(len(states), -1).astype(np.float64),
                EPSILON, 1.0 - EPSILON)
    return float((-(x * np.log(x) + (1.0 - x) * np.log(1.0 - x))).mean())


def reconstruction_scores(states, reconstruction):
    """`{BCE, MSE}` of a reconstruction, scored as the network scores itself:
    the prediction clipped to [1e-7, 1 - 1e-7], averaged over features and
    then over states. Added 2026-10-06, to score the test clips."""
    x = states.reshape(len(states), -1).astype(np.float64)
    y = reconstruction.reshape(len(reconstruction), -1).astype(np.float64)
    p = np.clip(y, EPSILON, 1.0 - EPSILON)
    bce = -(x * np.log(p) + (1.0 - x) * np.log(1.0 - p))
    return {"BCE": float(bce.mean()), "MSE": float(np.square(x - y).mean())}


def mean_baseline(train, val):
    """`{train_BCE, train_MSE, val_BCE, val_MSE, rule}` for the mean predictor,
    and the BCE floor of each split, `train_BCE_floor` and `val_BCE_floor`."""
    mean = train.reshape(len(train), -1).astype(np.float64).mean(axis=0)
    train_bce, train_mse = _scores(mean, train)
    val_bce, val_mse = _scores(mean, val)
    return {"train_BCE": train_bce, "train_MSE": train_mse,
            "val_BCE": val_bce, "val_MSE": val_mse,
            "train_BCE_floor": bce_floor(train),
            "val_BCE_floor": bce_floor(val),
            "rule": "always predict the mean training state, scored as the "
                    "network's BCE and MSE metrics"}
