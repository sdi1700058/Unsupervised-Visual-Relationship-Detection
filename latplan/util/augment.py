#!/usr/bin/env python3
"""Clip-consistent augmentation for video object states.

The point of augmenting here is to multiply training samples without
destroying what the planner half of the pipeline needs. That constraint
rules most augmentation out.

FOSAE learns a transition model from consecutive frames. If each object's
position is re-randomised independently per frame, consecutive frames stop
being causally related: the autoencoder gets more data to reconstruct, and
the transition model gets noise. So every transform here is applied
identically to every frame of a clip — the trajectory is moved, mirrored or
rescaled as a whole, and the motion inside it survives.

`per_frame_jitter` is the exception, and it is deliberately available: it is
the control that shows what independent per-frame randomisation does to
plannability. Do not use it to grow a training set you intend to plan on.

Each augmented copy gets its own video id, because `build_transitions` in
`sequential` mode pairs frames that share one. Without that, an original
frame would be paired with an augmented one and the transition would be a
teleport.
"""

import numpy as np


def _clip_boxes(boxes, width, height):
    """Keep boxes inside the canvas and non-degenerate."""
    boxes = boxes.copy()
    boxes[..., 0] = np.clip(boxes[..., 0], 0, width - 1)
    boxes[..., 2] = np.clip(boxes[..., 2], 0, width - 1)
    boxes[..., 1] = np.clip(boxes[..., 1], 0, height - 1)
    boxes[..., 3] = np.clip(boxes[..., 3], 0, height - 1)
    # A transform can invert an edge; put them back in order.
    x1 = np.minimum(boxes[..., 0], boxes[..., 2])
    x2 = np.maximum(boxes[..., 0], boxes[..., 2])
    y1 = np.minimum(boxes[..., 1], boxes[..., 3])
    y2 = np.maximum(boxes[..., 1], boxes[..., 3])
    return np.stack([x1, y1, x2, y2], axis=-1)


def _padding_mask(boxes):
    """True where a slot is real. Padded slots are all-zero boxes."""
    return boxes.reshape(boxes.shape[:-1] + (4,)).any(axis=-1)


def hflip(images, boxes, width, height):
    """Mirror the clip left-right. Physically valid: motion stays motion."""
    out_boxes = boxes.astype(np.float32).copy()
    real = _padding_mask(boxes)
    x1 = out_boxes[..., 0].copy()
    x2 = out_boxes[..., 2].copy()
    out_boxes[..., 0] = width - 1 - x2
    out_boxes[..., 2] = width - 1 - x1
    out_boxes[~real] = 0
    # The patch has to mirror too, or appearance and position disagree.
    return images[..., ::-1, :].copy(), _clip_boxes(out_boxes, width, height)


def translate(images, boxes, width, height, dx, dy):
    """Shift every box in the clip by the same offset.

    The patch content is left alone on purpose. Decoupling appearance from
    position is exactly what we want the model to be able to do.
    """
    out_boxes = boxes.astype(np.float32).copy()
    real = _padding_mask(boxes)
    out_boxes[..., [0, 2]] += dx
    out_boxes[..., [1, 3]] += dy
    out_boxes[~real] = 0
    return images, _clip_boxes(out_boxes, width, height)


def rescale(images, boxes, width, height, factor):
    """Zoom every box about the canvas centre by one factor for the clip."""
    out_boxes = boxes.astype(np.float32).copy()
    real = _padding_mask(boxes)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    out_boxes[..., [0, 2]] = cx + (out_boxes[..., [0, 2]] - cx) * factor
    out_boxes[..., [1, 3]] = cy + (out_boxes[..., [1, 3]] - cy) * factor
    out_boxes[~real] = 0
    return images, _clip_boxes(out_boxes, width, height)


def per_frame_jitter(images, boxes, width, height, rng, magnitude=0.1):
    """Move every object independently in every frame.

    This is the control arm. It breaks temporal coherence by construction,
    so a model trained on it should reconstruct as well as any other and
    plan considerably worse. If it does not, the transition model was not
    using temporal structure in the first place — which is itself the
    finding.
    """
    out_boxes = boxes.astype(np.float32).copy()
    real = _padding_mask(boxes)
    dx = rng.uniform(-magnitude * width, magnitude * width, size=boxes.shape[:-1])
    dy = rng.uniform(-magnitude * height, magnitude * height, size=boxes.shape[:-1])
    out_boxes[..., 0] += dx
    out_boxes[..., 2] += dx
    out_boxes[..., 1] += dy
    out_boxes[..., 3] += dy
    out_boxes[~real] = 0
    return images, _clip_boxes(out_boxes, width, height)


def _video_of(frame_id):
    return frame_id.rsplit("/", 1)[0]


def augment_dataset(images, bboxes, names, frame_ids, methods,
                    width, height, seed=0, copies=1):
    """Return the originals with augmented copies appended.

    Parameters
    ----------
    methods : list[str] — any of 'hflip', 'translate', 'rescale',
        'reverse', 'jitter'.
    copies  : int — how many randomised copies per method that takes a
        random parameter ('translate', 'rescale', 'jitter'). Deterministic
        methods ('hflip', 'reverse') always produce exactly one.

    Returns
    -------
    (images, bboxes, names, frame_ids) with the augmented clips appended.
    Augmented frames carry a video id suffixed '+<method><n>', so
    `sequential` transitions never cross between a clip and its copy.
    """
    rng = np.random.RandomState(seed)

    videos = {}
    for i, fid in enumerate(frame_ids):
        videos.setdefault(_video_of(fid), []).append(i)

    out_images = [images]
    out_boxes = [bboxes]
    out_names = list(names)
    out_fids = list(frame_ids)

    for method in methods:
        n_copies = 1 if method in ("hflip", "reverse") else copies
        for c in range(n_copies):
            tag = "+{}{}".format(method, c if n_copies > 1 else "")
            im_parts, bb_parts = [], []
            for vid, idx in videos.items():
                idx = sorted(idx)
                im = images[idx]
                bb = bboxes[idx]
                order = list(range(len(idx)))

                if method == "hflip":
                    im, bb = hflip(im, bb, width, height)
                elif method == "translate":
                    dx = rng.uniform(-0.15, 0.15) * width
                    dy = rng.uniform(-0.15, 0.15) * height
                    im, bb = translate(im, bb, width, height, dx, dy)
                elif method == "rescale":
                    im, bb = rescale(im, bb, width, height,
                                     rng.uniform(0.7, 1.3))
                elif method == "reverse":
                    order = order[::-1]
                    im, bb = im[order], bb[order]
                elif method == "jitter":
                    im, bb = per_frame_jitter(im, bb, width, height, rng)
                else:
                    raise ValueError("unknown augmentation {!r}".format(method))

                im_parts.append(im)
                bb_parts.append(bb.astype(bboxes.dtype))
                for step, orig in enumerate(order):
                    out_names.append(names[idx[orig]])
                    out_fids.append("{}{}/{}".format(vid, tag, step))

            out_images.append(np.concatenate(im_parts, axis=0))
            out_boxes.append(np.concatenate(bb_parts, axis=0))

    return (np.concatenate(out_images, axis=0),
            np.concatenate(out_boxes, axis=0),
            out_names,
            out_fids)
