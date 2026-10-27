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
    python3 tools/planner/plan_video.py latents.npz --method bfs

Contents of the export:

    latents             (n_frames, U*P) int8    encoded frames
    gt_boxes            (n_frames, n_obj, 4)    annotated boxes, canvas pixels
    decoded_boxes       (n_frames, n_obj, 4)    boxes the decoder gives for each
                                                latent, so scoring needs no decoder
    bindings            (n_frames, U, A)        object FOSAE attended to, per
                                                unit and argument slot
    binding_confidence  (n_frames, U, A)        the attention mass behind
                                                each binding above
    U, A, P, n_bits, model_name, frame_ids

decoded_boxes is what makes the decoder unnecessary later. Any latent a plan
visits is reachable from the training transitions, so it is one of the latents
already in this table, and we can look its boxes up instead of decoding.
Latents the plan invents that are not in the table fall back to the nearest
one by Hamming distance, and the export records how often that happened.
"""

import argparse
import sys
from pathlib import Path


def bindings_from_attention(attention):
    """Hard (unit, argument) to object bindings, and the attention behind each.

    FOSAE attends softly over the objects of a state. The hard binding is the
    argmax, which is what `latplan/util/fol.py` already does, and this is the
    same definition rather than a second one. `test_relation_inversion.py`
    pins the two together, because an export that disagreed with the oracle
    arm about what a bit means would make the two arms incomparable.

    attention is (batch, U, A, num_objs). The answer is (batch, U, A).
    """
    import numpy as np
    attention = np.asarray(attention)
    bindings = np.argmax(attention, axis=-1).astype(np.int32)
    confidences = np.max(attention, axis=-1).astype(np.float32)
    return bindings, confidences


def invert_bit(bit, bindings, P):
    """One bit of the U*P latent, as (object, ..., predicate).

    THIS IS WHY THE EXPORT NEEDED A `bindings` KEY. Without it a FOSAE bit
    inverts to (unit, predicate) and stops, so stage one of the two-stage
    evaluation the supervisors set on 2026-09-15 is not computable for any
    dataset. With it a bit reads as a predicate over named objects, which is
    the shape their record asks for.

    `bindings` is one state's (U, A) table. The returned tuple is the A bound
    object indices followed by the predicate index, so a binary predicate
    gives (subject, object, predicate) and a ternary one gives four entries.
    The arity is not fixed at two, because FOSAE's A is a parameter.

    The (unit, predicate) split matches `extract_predicate_values`, which
    reshapes a flat (U*P,) latent to (U, P). A different split here would
    name the wrong predicate for every bit.
    """
    import numpy as np
    bindings = np.asarray(bindings)
    units = bindings.shape[0]
    if bit < 0 or bit >= units * P:
        raise IndexError(
            "bit %r is outside a latent of %d units by %d predicates"
            % (bit, units, P))
    unit, predicate = divmod(int(bit), int(P))
    return tuple(int(o) for o in bindings[unit]) + (predicate,)


def relations_of_state(latent, bindings, P):
    """Every TRUE bit of one state, as a relation over objects.

    This is R(O) in the supervisors' own notation, from the record of
    2026-09-15: the estimated set of relations their matching metric scores
    against the true set R'(O).

    `latent` is one state's flat (U*P,) code, `bindings` its (U, A) table.
    The rounding matches `extract_predicate_values`, because a raw encoder
    output is not exactly 0 or 1 and reading 0.6 as false would drop
    relations without saying so.
    """
    import numpy as np
    latent = np.asarray(latent).reshape(-1)
    bindings = np.asarray(bindings)
    units = bindings.shape[0]
    if latent.size != units * P:
        raise ValueError(
            "latent of %d bits does not fit %d units by %d predicates"
            % (latent.size, units, P))
    truth = np.round(latent).astype(int)
    return [invert_bit(bit, bindings, P)
            for bit in range(latent.size) if truth[bit]]


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


def export(model_dir, npz_path=None, out_path=None, batch=512):
    import numpy as np

    from tools.planner.common.encode import (
        load_model, load_npz_states, encode_all, apply_in_batches)
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

    latents = encode_all(net, states, batch=batch)
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
    recon = apply_in_batches(net.autoencode, states, batch)
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

    # WHICH OBJECTS EACH BIT IS ABOUT, and without this the export is not
    # usable for stage one. The supervisors asked on 2026-09-15 for relations
    # as a list of predicates with objects as arguments. A latent bit alone
    # inverts to (unit, predicate) and stops there, so the FOSAE arm and the
    # oracle arm of `relation_oracle.py` could not be compared at all.
    #
    # `bindings` is (batch, U, A): for each state, each unit and each argument
    # slot, the index of the object FOSAE attended to. `invert_bit` above
    # turns a bit index into (subject, object, predicate) with it.
    #
    # `binding_confidence` ships beside it because the binding is an argmax
    # over a soft attention. A bit whose binding rests on 0.34 of the mass is
    # a different claim from one resting on 0.99, and a metric that cannot
    # see the difference cannot report it.
    attention = np.asarray(net.encode_attention(states))
    bindings, binding_confidence = bindings_from_attention(attention)
    payload["bindings"] = bindings
    payload["binding_confidence"] = binding_confidence
    print("bindings %s from attention %s"
          % (bindings.shape, attention.shape))

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
    ap.add_argument("--batch", type=int, default=512,
                    help="states per call to the model. The whole array in "
                         "one call was killed at 8 GB on the 14,314 states "
                         "of the KIT role clips, 2026-10-10. 0 means one "
                         "call.")
    args = ap.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    export(args.model_dir, args.npz_path, args.out, batch=args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
