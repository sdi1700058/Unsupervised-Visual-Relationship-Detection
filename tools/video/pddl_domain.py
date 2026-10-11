#!/usr/bin/env python3
"""The PDDL domain of Something-Else, read for the relations it states.

WHY IT IS HERE AND NOT IN `workbench`. The relations of Something-Else come
from the published domain of Migimatsu and Bohg, arXiv 2109.14718, shipped as
`data/video/something_else/raw/twentybn_domain.pddl`. Reading it is a step of
the conversion, so it belongs in the public tree with the converter that uses
it. `workbench/tools/something_else_relations.py` wrote these functions first,
on 2026-10-03, to measure the dataset, and now imports them from here rather
than holding a second copy. Plan
notes/plans/2026-10-10-relations-and-boxes.md, A2.

The domain states, for each of its 174 actions, a precondition and an effect
over 37 predicates, 10 of them binary. An action FLIPS a predicate when the
effect asserts what the precondition denies, or the reverse, which is the
domain's own statement that the action changes it. Measured 2026-10-11 over
the 234 clips that train: every clip's template maps to an action that flips
at least one binary predicate, 151 of those argument pairs are object to
object and 139 are object to hand, and none names anything that is not a
track of the clip.
"""

import re

# A declaration looks like "(above ?a - sth ?b - sth)". The arity is the count
# of "?" variables, so a binary predicate names two.
_DECL = re.compile(r"\(([a-z][a-z0-9-]*)((?:\s+\?[a-z0-9]+\s*-\s*[a-z]+)+)\)")
_VAR = re.compile(r"\?[a-z0-9]+")

# A static property cannot be a transition: an object is bendable or it is not.
STATIC_PREFIX = "is-"


def parse_domain(path):
    """(predicates, actions) from a PDDL domain.

    `predicates` maps a name to its arity. `actions` maps a name to
    (precondition_text, effect_text).
    """
    with open(path) as handle:
        text = handle.read()

    predicates = {}
    block = text.split("(:predicates", 1)
    if len(block) > 1:
        # The block ends at the first line holding only a closing paren.
        body = re.split(r"\n\s*\)\s*\n", block[1], 1)[0]
        for name, args in _DECL.findall(body):
            predicates[name] = len(_VAR.findall(args))
        # A nullary or already-seen name can also appear without arguments.
        for name in re.findall(r"\(([a-z][a-z0-9-]*)\s*\)", body):
            predicates.setdefault(name, 0)

    actions = {}
    for chunk in text.split("(:action ")[1:]:
        name = chunk.split(None, 1)[0]
        pre = chunk.split(":precondition", 1)
        eff = chunk.split(":effect", 1)
        actions[name] = (pre[1].split(":effect")[0] if len(pre) > 1 else "",
                         eff[1] if len(eff) > 1 else "")
    return predicates, actions


def _mentions(text, name):
    return re.search(r"\(\s*%s[\s)]" % re.escape(name), text) is not None


def _negated(text, name):
    return re.search(r"\(\s*not\s*\(\s*%s[\s)]" % re.escape(name),
                     text) is not None


def flips(actions, predicates, binary_only=False):
    """Per action, the predicates the published domain says it CHANGES.

    A predicate flips when the effect asserts it and the precondition denies
    it, or the reverse. That is the domain's own statement about a transition,
    so no threshold of this project's enters it.
    """
    out = {}
    for action, (pre, eff) in actions.items():
        moved = set()
        for name, arity in predicates.items():
            if name.startswith(STATIC_PREFIX):
                continue
            if binary_only and arity < 2:
                continue
            if not _mentions(eff, name):
                continue
            # The effect touches it. It is a flip unless the precondition
            # already asserts it in the same polarity.
            if _negated(eff, name) != _negated(pre, name) or \
                    not _mentions(pre, name):
                moved.add(name)
        out[action] = moved
    return out


def template_ids(text):
    """`{action: template id}` from the `; <id> <template>` line above each
    action, the id being the template's id in SSv2's labels.json."""
    out = {}
    for m in re.finditer(r";\s*(\d+)\s[^\n]*\n\s*\(:action\s+(\S+)", text):
        out[m.group(2)] = m.group(1)
    return out


def flipping_template_ids(path):
    """Sorted template ids whose action moves a two-object predicate, by the
    published domain's own precondition against its own effect."""
    with open(path) as handle:
        text = handle.read()
    predicates, actions = parse_domain(path)
    ids = template_ids(text)
    moved = flips(actions, predicates, binary_only=True)
    return sorted((ids[a] for a, names in moved.items()
                   if names and a in ids), key=int)


def binary_args(actions, action, name):
    """`{(a, b)}`: the argument pairs `name` is used with in `action`.

    Both halves are read, because a flip names the predicate in the
    precondition and in the effect, and either may carry the arguments. An
    argument is a placeholder such as `?a` or a constant such as `hand`,
    which the domain declares at `(:constants hand - object)`. The converter
    binds a placeholder by position and `hand` to the hand track.
    """
    if action not in actions:
        return set()
    out = set()
    for half in actions[action]:
        for m in re.finditer(r"\(\s*%s\s+([^()]*)\)" % re.escape(name), half):
            parts = m.group(1).split()
            if len(parts) == 2:
                out.add((parts[0], parts[1]))
    return out

