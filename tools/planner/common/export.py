"""Read a planner export and answer the two questions scoring asks.

An export holds the encoded frames, the annotated boxes and the boxes the
decoder produced for each frame. That is enough to score a plan without
loading keras, which is the whole point: the model lives on the cluster, the
planner can run on a laptop.

The one thing an export cannot do is decode a latent that is not in it. A
plan built from the training deltas can only reach states the model reached,
so in practice every latent on a plan is already in the table. When one is
not, we fall back to the nearest row by Hamming distance and count it, so a
run that leans on the fallback is visible rather than silent.
"""

from pathlib import Path


class Export:

    def __init__(self, path):
        import numpy as np

        self.path = Path(path)
        # Everything in an export is a plain numeric or unicode array, so
        # pickle stays off.
        data = np.load(self.path)

        self.latents = data["latents"].astype(np.int8)
        self.gt_boxes = data["gt_boxes"].astype(np.float32)
        self.decoded_boxes = data["decoded_boxes"].astype(np.float32)
        self.n_bits = int(data["n_bits"])
        self.model_name = str(data["model_name"])
        self.actions = data["actions"] if "actions" in data.files else None

        self.parameters = {k: int(data[k]) for k in ("U", "A", "P")
                           if k in data.files}

        # Row lookup by exact latent, so the common case costs nothing.
        self._index = {row.tobytes(): i for i, row in enumerate(self.latents)}
        self.fallback_count = 0

    def __len__(self):
        return len(self.latents)

    def boxes_for(self, latent):
        """Boxes for one latent. Exact hit if possible, nearest row if not."""
        import numpy as np

        latent = np.asarray(latent, dtype=np.int8).reshape(-1)
        hit = self._index.get(latent.tobytes())
        if hit is not None:
            return self.decoded_boxes[hit]

        self.fallback_count += 1
        distances = (self.latents ^ latent).sum(axis=1)
        return self.decoded_boxes[int(np.argmin(distances))]

    def boxes_for_trace(self, trace):
        import numpy as np
        return np.stack([self.boxes_for(z) for z in np.asarray(trace)])

    def transitions(self):
        """Return (pre, suc) for the pddl method.

        Uses actions.csv when the export carried it, and consecutive frames
        otherwise.
        """
        if self.actions is not None:
            half = self.actions.shape[1] // 2
            return self.actions[:, :half], self.actions[:, half:]
        return self.latents[:-1], self.latents[1:]


def load(path):
    return Export(path)
