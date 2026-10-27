#!/usr/bin/env python3
"""A FOSAE bit must invert to (subject, object, predicate), not to (unit, predicate).

The supervisors asked for relations as "a list of predicates with objects as
arguments", in the meeting of 2026-09-15. Two arms of this project produce a
bit, and until now only one of them could say which objects a bit is about.

`planner/relation_oracle.py` writes `predicates` and `num_objs`, so an
oracle bit inverts to (subject, object, predicate).
`planner/export_latents.py` wrote no `bindings` key, so a FOSAE bit
inverted to (unit, predicate) and stopped. That is why stage one of the
two-stage evaluation was not computable for any dataset, whatever the dataset
shipped.

The producers already existed. `model/model.py` defines `encode_attention`,
and `model/util/fol.py` turns that attention into hard bindings by argmax.
Neither was wired into the export.

These tests are pure. They touch no model and no keras, so they run in the
public suite in milliseconds.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

from planner.export_latents import (  # noqa: E402
    bindings_from_attention, invert_bit)


class TestBindingsFromAttention(unittest.TestCase):

    def test_the_hard_binding_is_the_argmax_object(self):
        # (batch=1, U=1, A=2, num_objs=3)
        attention = np.array([[[[0.1, 0.7, 0.2],
                                [0.6, 0.3, 0.1]]]])
        bindings, confidences = bindings_from_attention(attention)
        np.testing.assert_array_equal(bindings, [[[1, 0]]])
        np.testing.assert_allclose(confidences, [[[0.7, 0.6]]])

    def test_the_shape_drops_the_object_axis(self):
        attention = np.zeros((4, 3, 2, 7))
        bindings, confidences = bindings_from_attention(attention)
        self.assertEqual(bindings.shape, (4, 3, 2))
        self.assertEqual(confidences.shape, (4, 3, 2))

    def test_it_agrees_with_the_producer_in_latplan(self):
        """The export must not invent a second definition of a binding.

        `model/util/fol.py` already computes this, and a copy that drifts
        from it would make the oracle arm and the FOSAE arm disagree about
        what a bit means. This pins them together.
        """
        from model.util.fol import extract_attention_bindings
        rng = np.random.RandomState(0)
        attention = rng.rand(5, 4, 2, 6)
        mine = bindings_from_attention(attention)
        theirs = extract_attention_bindings(attention)
        np.testing.assert_array_equal(mine[0], theirs[0])
        np.testing.assert_allclose(mine[1], theirs[1])

    def test_an_integer_dtype_survives_the_round_trip(self):
        # The export saves these to npz, and a float index is not an index.
        attention = np.zeros((1, 1, 2, 3))
        attention[0, 0, 0, 2] = 1.0
        attention[0, 0, 1, 1] = 1.0
        bindings, _ = bindings_from_attention(attention)
        self.assertTrue(np.issubdtype(bindings.dtype, np.integer))


class TestInvertBit(unittest.TestCase):
    """One bit of the U*P latent, read back as a relation over objects."""

    def setUp(self):
        # U=2 units, A=2 arguments, P=3 predicates per unit.
        # Unit 0 binds objects (5, 6). Unit 1 binds objects (7, 8).
        self.bindings = np.array([[5, 6], [7, 8]])

    def test_the_first_bit_is_predicate_zero_of_unit_zero(self):
        got = invert_bit(0, self.bindings, P=3)
        self.assertEqual(got, (5, 6, 0))

    def test_a_bit_inside_the_first_unit_keeps_its_objects(self):
        self.assertEqual(invert_bit(2, self.bindings, P=3), (5, 6, 2))

    def test_crossing_into_the_next_unit_changes_the_objects(self):
        # bit 3 is unit 1, predicate 0
        self.assertEqual(invert_bit(3, self.bindings, P=3), (7, 8, 0))

    def test_the_layout_matches_the_reshape_the_project_already_uses(self):
        """`extract_predicate_values` reshapes (U*P,) to (U, P).

        If this function disagreed with that reshape, every inverted bit
        would name the wrong predicate. So the layout is pinned against the
        producer rather than restated.
        """
        from model.util.fol import extract_predicate_values
        U, P = 2, 3
        flat = np.arange(U * P).reshape(1, U * P)
        table = extract_predicate_values(flat, U, P)
        for bit in range(U * P):
            subject, obj, predicate = invert_bit(bit, self.bindings, P=P)
            unit = bit // P
            self.assertEqual(table[0, unit, predicate], bit)
            self.assertEqual((subject, obj), tuple(self.bindings[unit]))

    def test_a_bit_past_the_end_is_refused(self):
        with self.assertRaises(IndexError):
            invert_bit(6, self.bindings, P=3)

    def test_a_negative_bit_is_refused_rather_than_wrapping(self):
        # -1 would silently index the last unit, which is a wrong answer
        # rather than an error.
        with self.assertRaises(IndexError):
            invert_bit(-1, self.bindings, P=3)

    def test_more_than_two_arguments_returns_the_whole_tuple(self):
        """A ternary predicate is a tuple, not a pair.

        FOSAE's A is a parameter, and the supervisors asked for predicates
        with objects as arguments without fixing the arity at two.
        """
        bindings = np.array([[1, 2, 3]])
        self.assertEqual(invert_bit(0, bindings, P=2), (1, 2, 3, 0))


class TestRelationsOfState(unittest.TestCase):
    """Every TRUE bit of one state, as relations over objects.

    This is the whole point of the `bindings` key: one call turns a state's
    latent into the set the supervisors' matching metric consumes, which
    their record calls R(O).
    """

    def setUp(self):
        from planner.export_latents import relations_of_state
        self.relations_of_state = relations_of_state
        # U=2, A=2, P=3
        self.bindings = np.array([[5, 6], [7, 8]])

    def test_only_the_true_bits_become_relations(self):
        latent = np.array([1, 0, 0,  0, 0, 1])
        got = self.relations_of_state(latent, self.bindings, P=3)
        self.assertEqual(got, [(5, 6, 0), (7, 8, 2)])

    def test_an_all_false_state_has_no_relations(self):
        latent = np.zeros(6, dtype=int)
        self.assertEqual(
            self.relations_of_state(latent, self.bindings, P=3), [])

    def test_an_all_true_state_gives_one_relation_for_each_bit(self):
        latent = np.ones(6, dtype=int)
        got = self.relations_of_state(latent, self.bindings, P=3)
        self.assertEqual(len(got), 6)
        self.assertEqual(got[0], (5, 6, 0))
        self.assertEqual(got[-1], (7, 8, 2))

    def test_a_float_latent_is_rounded_the_way_the_project_rounds(self):
        """`extract_predicate_values` rounds before it reshapes.

        A raw encoder output is not exactly 0 or 1, and reading 0.6 as false
        would silently drop relations.
        """
        latent = np.array([0.6, 0.4, 0.0,  0.0, 0.0, 0.51])
        got = self.relations_of_state(latent, self.bindings, P=3)
        self.assertEqual(got, [(5, 6, 0), (7, 8, 2)])

    def test_a_latent_that_does_not_fit_the_bindings_is_refused(self):
        # 7 bits against 2 units of 3 predicates is a mismatch, and a silent
        # truncation here would produce a plausible wrong answer.
        with self.assertRaises(ValueError):
            self.relations_of_state(np.zeros(7), self.bindings, P=3)


class TestTheCommand(unittest.TestCase):
    """`show_relations.py` is the one command the DONE WHEN clause asks for.

    It reads an export and prints each state's relations as
    (subject, object, predicate). It needs numpy alone, so it runs anywhere
    the export does, including Sherlock's Python 3.6.
    """

    def setUp(self):
        import shutil
        import tempfile
        from planner import show_relations
        self.show_relations = show_relations
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root)

    def _export(self, **over):
        payload = {
            "latents": np.array([[1, 0, 0, 0, 0, 1],
                                 [0, 0, 0, 0, 0, 0]], dtype=np.int8),
            "bindings": np.array([[[5, 6], [7, 8]],
                                  [[5, 6], [7, 8]]], dtype=np.int32),
            "binding_confidence": np.full((2, 2, 2), 0.9, dtype=np.float32),
            "U": 2, "A": 2, "P": 3,
            "frame_ids": np.asarray(["f0", "f1"], dtype="U256"),
        }
        payload.update(over)
        path = os.path.join(self.root, "planner_export.npz")
        np.savez_compressed(path, **payload)
        return path

    def test_it_prints_one_line_for_each_true_bit(self):
        out = self.show_relations.render(self._export())
        self.assertIn("(5, 6, 0)", out)
        self.assertIn("(7, 8, 2)", out)

    def test_it_names_the_frame_each_relation_came_from(self):
        out = self.show_relations.render(self._export())
        self.assertIn("f0", out)

    def test_a_state_with_no_true_bit_says_so_rather_than_vanishing(self):
        """Silence and an empty answer must not look the same.

        `unittest discover` over an empty directory printing OK is the same
        failure one level up, and probes.json records that this repository
        has been burned by it.
        """
        out = self.show_relations.render(self._export())
        self.assertIn("f1", out)
        self.assertIn("no relation", out.lower())

    def test_an_export_without_bindings_is_refused_by_name(self):
        """The whole point is the missing key, so the message says which.

        Every export written before 2026-09-21 lacks it, and a reader who
        meets one needs to know to re-export rather than to debug a metric.
        """
        path = self._export()
        data = dict(np.load(path, allow_pickle=False))
        del data["bindings"]
        os.remove(path)
        np.savez_compressed(path, **data)
        with self.assertRaises(KeyError) as caught:
            self.show_relations.render(path)
        self.assertIn("bindings", str(caught.exception))

    def test_the_confidence_is_reported_when_it_is_asked_for(self):
        out = self.show_relations.render(self._export(), with_confidence=True)
        self.assertIn("0.9", out)

    def test_one_frame_can_be_selected(self):
        out = self.show_relations.render(self._export(), frame="f1")
        self.assertIn("f1", out)
        self.assertNotIn("(5, 6, 0)", out)


if __name__ == "__main__":
    unittest.main()
