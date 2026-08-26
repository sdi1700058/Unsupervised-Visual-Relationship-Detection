#!/usr/bin/env python3
"""tools/planner/common/decode.py — shared latent -> feature vector -> bbox.

The FirstOrderSAE decoder outputs a feature vector of the same layout as
`blocks_activation` domain (see model.py :: labeled_objects_activation +
blocks_activation): `[patch_size**2 * 3 | x1_onehot(X) | y1_onehot(Y) | x2_onehot(X) | y2_onehot(Y)]`.

This module extracts the bbox portion and converts the onehot bins back to
pixel coordinates.
"""


def _picsize_grid():
    import numpy as np
    from latplan.puzzles.puzzle_labeled_objects import PICSIZE
    return (np.array(PICSIZE) // 5).astype(int)


def _feat_layout(feat_dim):
    """Return (patch_len, X, Y) given feat_dim.

    feat_dim = patch_side**2 * 3 + 2*X + 2*Y
    """
    grid = _picsize_grid()
    Y, X = int(grid[0]), int(grid[1])
    bbox_len = 2 * X + 2 * Y
    patch_len = feat_dim - bbox_len
    return patch_len, X, Y


def decode_latent(ae, z):
    """Decode one binary latent to the reconstructed feature vector.

    Parameters
    ----------
    ae : loaded FirstOrderSAE.
    z  : np.ndarray (U*P,) int  OR  (batch, U*P).

    Returns
    -------
    features : np.ndarray (batch, num_objs, feat_dim) float
    """
    import numpy as np
    z = np.asarray(z)
    if z.ndim == 1:
        z = z[None, :]
    z_float = z.astype(np.float32)
    recon = ae.decode(z_float)
    return np.asarray(recon)


def features_to_bboxes(features):
    """Convert decoded features to per-object bboxes in canvas pixel coords.

    Parameters
    ----------
    features : np.ndarray (batch, num_objs, feat_dim) — output of `decode_latent`.

    Returns
    -------
    bboxes : np.ndarray (batch, num_objs, 4) float — (x1, y1, x2, y2) in
             canvas pixel coords (CANVAS via PICSIZE).
    """
    import numpy as np
    from latplan.puzzles.puzzle_labeled_objects import PICSIZE

    batch, num_objs, feat_dim = features.shape
    patch_len, X, Y = _feat_layout(feat_dim)
    bbox_start = patch_len

    # Slice out the four onehot blocks: x1, y1, x2, y2.
    x1_oh = features[:, :, bbox_start:bbox_start + X]
    y1_oh = features[:, :, bbox_start + X:bbox_start + X + Y]
    x2_oh = features[:, :, bbox_start + X + Y:bbox_start + 2 * X + Y]
    y2_oh = features[:, :, bbox_start + 2 * X + Y:bbox_start + 2 * X + 2 * Y]

    # argmax → bin index → pixel via linear scaling to PICSIZE.
    canvas_h, canvas_w = int(PICSIZE[0]), int(PICSIZE[1])
    x1 = np.argmax(x1_oh, axis=-1) * (canvas_w / X)
    y1 = np.argmax(y1_oh, axis=-1) * (canvas_h / Y)
    x2 = np.argmax(x2_oh, axis=-1) * (canvas_w / X)
    y2 = np.argmax(y2_oh, axis=-1) * (canvas_h / Y)

    return np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)


def decode_trace_to_bboxes(ae, z_trace):
    """Convenience: decode a plan trace (list/array of latents) to bboxes.

    Parameters
    ----------
    ae      : loaded FirstOrderSAE.
    z_trace : np.ndarray (T, U*P) — one latent per plan step.

    Returns
    -------
    bboxes : np.ndarray (T, num_objs, 4) float pixel coords.
    """
    import numpy as np
    z_trace = np.asarray(z_trace)
    features = decode_latent(ae, z_trace)         # (T, num_objs, feat_dim)
    return features_to_bboxes(features)
