"""Tests for tools/datasets/pddl_domain.py.

WHY PROPERTIES AND NOT GOLDEN NUMBERS. The real domain file is data and not
code, so an assertion of 174 actions would fail on a machine without it.
These pin the reading of PDDL the conversion depends on: that arity decides
what counts as a relation between two objects, that a static property is
never a transition, and that a flip needs the effect and the precondition to
disagree.

Moved here from `workbench/tests/` on 2026-10-11 with the code
they cover, because the converter is public and may not import `workbench`.
"""

import os
import tempfile
import unittest

from tools.datasets import pddl_domain as se


DOMAIN = """(define (domain twentybn)
    (:requirements :strips :typing)
    (:types sth object)
    (:predicates
        ; static properties
        (is-rigid ?a - sth)
        ; mutable properties
        (broken ?a - sth)
        (nextto ?a - sth ?b - sth)
        (touching ?a - object ?b - object)
        (on ?a - sth ?b - sth)
    )

    (:action move-together
        :parameters (?a - sth ?b - sth)
        :precondition (and (not (nextto ?a ?b)) (is-rigid ?a))
        :effect (nextto ?a ?b)
    )
    (:action break-it
        :parameters (?a - sth)
        :precondition (not (broken ?a))
        :effect (broken ?a)
    )
    (:action hold-still
        :parameters (?a - sth ?b - sth)
        :precondition (on ?a ?b)
        :effect (on ?a ?b)
    )
    (:action declare-rigid
        :parameters (?a - sth)
        :precondition (not (is-rigid ?a))
        :effect (is-rigid ?a)
    )
    (:action last-one
        :parameters (?a - sth ?b - sth)
        :precondition (nextto ?a ?b)
        :effect (and
            ; (on ?a ?b)
            (touching ?a ?b)
        )
    )
    (:axiom
        :vars (?a - sth ?b - sth)
        :context (on ?a ?b)
        :implies (not (broken ?a))
    )
)
"""


def _domain_file():
    handle = tempfile.NamedTemporaryFile("w", suffix=".pddl", delete=False)
    handle.write(DOMAIN)
    handle.close()
    return handle.name


class ParseDomain(unittest.TestCase):
    def setUp(self):
        self.path = _domain_file()
        self.predicates, self.actions = se.parse_domain(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_every_declared_predicate_is_found(self):
        self.assertEqual(set(self.predicates),
                         {"is-rigid", "broken", "nextto", "touching", "on"})

    def test_arity_separates_object_to_object_from_a_property(self):
        """The binary subset is the only one this project can use as a pair."""
        self.assertEqual(self.predicates["broken"], 1)
        self.assertEqual(self.predicates["nextto"], 2)
        self.assertEqual(self.predicates["touching"], 2)
        binary = [n for n, a in self.predicates.items() if a >= 2]
        self.assertEqual(sorted(binary), ["nextto", "on", "touching"])

    def test_every_action_is_found_with_both_halves(self):
        self.assertEqual(set(self.actions),
                         {"move-together", "break-it", "hold-still",
                          "declare-rigid", "last-one"})
        for pre, eff in self.actions.values():
            self.assertTrue(pre.strip())
            self.assertTrue(eff.strip())

    def test_the_precondition_does_not_swallow_the_effect(self):
        pre, eff = self.actions["move-together"]
        self.assertIn("is-rigid", pre)
        self.assertNotIn(":effect", pre)
        self.assertIn("nextto", eff)


class Flips(unittest.TestCase):
    def setUp(self):
        self.path = _domain_file()
        self.predicates, self.actions = se.parse_domain(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_a_negated_precondition_becoming_an_asserted_effect_is_a_flip(self):
        moved = se.flips(self.actions, self.predicates)
        self.assertIn("nextto", moved["move-together"])

    def test_an_unchanged_predicate_is_not_a_flip(self):
        """`hold-still` asserts `on` in both halves, so nothing moved."""
        moved = se.flips(self.actions, self.predicates)
        self.assertEqual(moved["hold-still"], set())

    def test_a_static_property_is_never_a_flip(self):
        """An object is rigid or it is not. That is not a transition.

        `declare-rigid` flips `is-rigid` in the PDDL and must still report
        nothing, because a static property cannot be a state change.
        """
        moved = se.flips(self.actions, self.predicates)
        self.assertEqual(moved["declare-rigid"], set())

    def test_binary_only_drops_the_single_argument_predicates(self):
        every = se.flips(self.actions, self.predicates)
        binary = se.flips(self.actions, self.predicates, binary_only=True)
        self.assertIn("broken", every["break-it"])
        self.assertEqual(binary["break-it"], set())
        self.assertIn("nextto", binary["move-together"])

    def test_a_prefix_does_not_match_a_longer_name(self):
        """`(on ?a ?b)` must not be found inside `(onsurface ?a)`.

        The real domain declares both, so a substring search would report
        `on` as flipping on every action that touches `onsurface`.
        """
        self.assertFalse(se._mentions("(onsurface ?a)", "on"))
        self.assertTrue(se._mentions("(on ?a ?b)", "on"))


class TemplateIds(unittest.TestCase):
    """Which SSv2 template ids the training clips may come from.

    Added 2026-10-03 for plan train-the-rungs step 1.2b. The published domain
    writes `; <id> <template>` above each action, and `<id>` is the template's
    id in SSv2's labels.json, so a clip whose template's action moves a
    two-object predicate can be chosen by id.
    """

    TEXT = """(define (domain twentybn)
    (:predicates
        (on ?a - sth ?b - sth)
        (open ?a - sth)
    )

    ; 0 Opening something
    (:action open-it
        :parameters (?a)
        :precondition (not (open ?a))
        :effect (open ?a))
    ; 1 Putting something on something
    (:action put-on
        :parameters (?a ?b)
        :precondition (not (on ?a ?b))
        :effect (on ?a ?b))
)
"""

    def test_each_action_takes_the_id_written_above_it(self):
        self.assertEqual(se.template_ids(self.TEXT),
                         {"open-it": "0", "put-on": "1"})

    def test_only_actions_moving_a_two_object_predicate_are_listed(self):
        with tempfile.NamedTemporaryFile("w", suffix=".pddl",
                                         delete=False) as handle:
            handle.write(self.TEXT)
        try:
            self.assertEqual(se.flipping_template_ids(handle.name), ["1"])
        finally:
            os.remove(handle.name)


if __name__ == "__main__":
    unittest.main()


class BinaryArgs(unittest.TestCase):
    """The argument pairs a flipped predicate is used with, so the converter
    knows which tracks the relation joins. Added 2026-10-11, plan A2."""

    def setUp(self):
        self.path = _domain_file()
        self.predicates, self.actions = se.parse_domain(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_the_argument_pair_of_a_two_placeholder_predicate(self):
        self.assertEqual(
            se.binary_args(self.actions, "move-together", "nextto"),
            {("?a", "?b")})

    def test_a_predicate_the_action_does_not_name_gives_nothing(self):
        self.assertEqual(
            se.binary_args(self.actions, "break-it", "nextto"), set())

    def test_the_effect_and_the_precondition_are_both_read(self):
        self.assertEqual(se.binary_args(self.actions, "hold-still", "on"),
                         {("?a", "?b")})

    def test_an_unknown_action_gives_nothing(self):
        self.assertEqual(se.binary_args(self.actions, "no-such", "on"), set())


class CommentsAndBoundaries(unittest.TestCase):
    """Two defects an adversary found on 2026-10-11, both in the real
    domain. `twentybn_domain.pddl:997` holds `; (attached ?a ?b)`, a line
    its own author disabled, and reading it as true put a relation that
    does not exist into 5 of the 234 training clips. And the last action of
    the file, `wipe`, had an effect of 8,130 characters because the parser
    had no terminator and swallowed every axiom after it."""

    def setUp(self):
        self.path = _domain_file()
        self.predicates, self.actions = se.parse_domain(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_a_commented_out_line_is_not_part_of_the_effect(self):
        self.assertNotIn("on", se.binary_args(self.actions, "last-one", "on"))
        self.assertEqual(se.binary_args(self.actions, "last-one", "on"), set())

    def test_the_last_action_does_not_swallow_what_follows_it(self):
        self.assertNotIn(":axiom", self.actions["last-one"][1])
        self.assertLess(len(self.actions["last-one"][1]), 200)

    def test_the_last_action_keeps_its_own_effect(self):
        self.assertEqual(se.binary_args(self.actions, "last-one", "touching"),
                         {("?a", "?b")})


if __name__ == "__main__":
    unittest.main()
