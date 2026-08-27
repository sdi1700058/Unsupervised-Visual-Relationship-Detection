#!/usr/bin/env python3
"""Dump a trained model down to the few arrays the planner actually needs.

The planner needs three things: the latent code of every frame, the annotated
boxes, and the decoder. Encoding needs keras and the model. Searching and
scoring need neither.

So run this once where the model lives, then run the planner anywhere. The
export is small, a few hundred kilobytes for a whole clip, so it travels over
scp without thought.

    # on Sherlock, inside the venv
    python3 tools/planner/export_latents.py <model_dir> -o latents.npz

    # anywhere, no keras
    python3 tools/planner/plan_video.py --latents latents.npz --method bfs

Contents of the export:

    latents        (n_frames, U*P) int8    encoded frames
    gt_boxes       (n_frames, n_obj, 4)    annotated boxes, canvas pixels
    decoded_boxes  (n_frames, n_obj, 4)    boxes the decoder gives for each
                                           latent, so scoring needs no decoder
    U, A, P, n_bits, model_name, npz_path, frame_ids

decoded_boxes is what makes the decoder unnecessary later. Any latent a plan
visits is reachable from the training transitions, so it is one of the latents
already in this table, and we can look its boxes up instead of decoding.
Latents the plan invents that are not in the table fall back to the nearest
one by Hamming distance, and the export records how often that happened.
"""

import argparse
import sys
from pathlib import Path


def _dedupe_transitions(rows):
    """Keep one row per distinct (add, delete) effect pair.

    Two transitions that flip the same bits the same way describe the same
    operator, so the second one adds nothing a planner can use. Rows where
    nothing changes go too: they would give the search a free self-loop.

    Order is preserved, so the export stays reproducible.
    """
    import numpy as np

    half = rows.shape[1] // 2
    pre, suc = rows[:, :half], rows[:, half:]

    diff = suc.astype(np.int16) - pre.astype(np.int16)
    signature = np.ascontiguousarray(
        np.concatenate([(diff > 0), (diff < 0)], axis=1).astype(np.int8))

    view = signature.view(
        np.dtype((np.void, signature.dtype.itemsize * signature.shape[1])))
    _, first = np.unique(view, return_index=True)

    keep = np.sort(first)
    keep = keep[diff[keep].any(axis=1)]
    return rows[keep]


def _assert_initialized():
    """Fail loudly when the session still holds uninitialized variables.

    Initializing them here would be worse than crashing: the weights would be
    replaced by random values and the export would look fine while being
    meaningless. So report and stop.
    """
    try:
        import keras.backend as K
        import tensorflow as tf
    except ImportError:
        return
    names = K.get_session().run(tf.report_uninitialized_variables())
    if len(names):
        shown = ", ".join(n.decode() if isinstance(n, bytes) else str(n)
                          for n in names[:6])
        raise SystemExit(
            f"{len(names)} uninitialized variables after loading the model "
            f"({shown}...). A TensorFlow session was created after the "
            "weights were loaded. Check the import order in export(): the "
            "data must load before the model.")


def export(model_dir, npz_path=None, out_path=None):
    import numpy as np

    from tools.planner.common.encode import (
        load_model, load_npz_states, encode_all)
    from tools.planner.common.decode import features_to_bboxes

    model_dir = Path(model_dir).resolve()

    # Load the data BEFORE the model, and do not reorder these two.
    # load_npz_states imports strips, which imports config, and config.py
    # calls load_session() at module level. Creating a TensorFlow session
    # after the weights are already loaded leaves the new session with
    # uninitialized variables, and predict then dies with
    #   FailedPreconditionError: Attempting to use uninitialized value
    #                            temperature_1
    # Loading the data first means the session exists before the model does.
    states, gt_boxes, _names, frame_ids = load_npz_states(model_dir, npz_path)
    print(f"{len(states)} frames, {states.shape[1]} object slots")

    print(f"loading {model_dir}")
    net = load_model(model_dir)
    _assert_initialized()

    latents = encode_all(net, states)
    print(f"latents {latents.shape}, {int(latents.sum())} bits set")

    # Decode every frame now, so the scoring stage never needs the decoder.
    #
    # Via autoencode rather than decode. FirstOrderSAE's standalone decoder is
    # not a self-contained latent-to-output graph: it still references the
    # autoencoder's input placeholder, so decoder.predict(z) dies with
    #   InvalidArgumentError: You must feed a value for placeholder tensor
    #                         'autoencoder' with dtype float and shape [?,3,392]
    # Fixing that would mean editing latplan/model.py, which SPEC C2 forbids.
    #
    # It is also unnecessary. Every latent in this table came from the state
    # at the same index, so decode(encode(x)) is autoencode(x) and the boxes
    # are identical. The planner never decodes a latent that is not already in
    # the table either — common/export.py looks boxes up by latent and falls
    # back to the nearest row by Hamming distance.
    recon = np.asarray(net.autoencode(states))
    decoded = features_to_bboxes(recon)
    print(f"decoded boxes {decoded.shape}")

    payload = {
        "latents": latents.astype(np.int8),
        "gt_boxes": np.asarray(gt_boxes, dtype=np.float32),
        "decoded_boxes": np.asarray(decoded, dtype=np.float32),
        "U": net.parameters["U"],
        "A": net.parameters["A"],
        "P": net.parameters["P"],
        "n_bits": latents.shape[1],
        "model_name": model_dir.name,
        # Plain unicode array, not object dtype, so the reader never needs
        # allow_pickle.
        "frame_ids": np.asarray([str(f) for f in frame_ids], dtype="U256"),
    }

    # actions.csv carries the transitions the model trained on. Every planner
    # reduces them to distinct effect pairs before use, so we do that here and
    # ship only the survivors. An all_pairs run has tens of thousands of rows
    # that collapse to a few hundred operators, which is the difference
    # between a file you can move around and one you cannot.
    actions_csv = model_dir / "actions.csv"
    if actions_csv.exists():
        rows = np.loadtxt(str(actions_csv), dtype=np.int8)
        if rows.ndim == 1:
            rows = rows[None, :]
        kept = _dedupe_transitions(rows)
        payload["actions"] = kept
        print(f"actions.csv: {len(rows)} transitions -> {len(kept)} distinct")
    else:
        print("no actions.csv; the planners will fall back to frame pairs")

    if out_path is None:
        out_path = model_dir / "planner_export.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)

    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.0f} KB)")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--npz-path", type=Path,
                    help="defaults to loaded_videos.json['npz_path']")
    ap.add_argument("-o", "--out", type=Path,
                    help="defaults to <model_dir>/planner_export.npz")
    args = ap.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    export(args.model_dir, args.npz_path, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
