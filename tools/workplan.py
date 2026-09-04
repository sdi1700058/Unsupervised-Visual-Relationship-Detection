#!/usr/bin/env python3
"""The work plan: units, assumptions, claims, and the arithmetic that grades them.

The design is `notes/docs/DESIGN_WORKPLAN.md`. The one rule that makes the rest
safe:

> **A score never decides anything. Its job is to localise disagreement.**

The value is not the number. It is that "this dataset feels better" becomes
"you weighted structure at 0.30 and I would weight it 0.20" — a specific,
checkable disagreement. So every number traces to a stored field, and **no score
is ever stored**: `e`, `support`, `net`, `contested`, `licence` and the wording
tier are all computed on read. There is no stored value to quietly adjust, and
moving a score requires adding an observation, which requires a command and an
output path.

    python3 tools/workplan.py next
    python3 tools/workplan.py render
    python3 tools/workplan.py check
    python3 tools/workplan.py score
    python3 tools/workplan.py contradictions
    python3 tools/workplan.py report

Standard library only, Python 3.6 clean.
"""

import argparse
import json
import math
import os
import sys


PLAN_PATH = os.environ.get("WORKPLAN", "notes/WORKPLAN.json")

BOARD_PATH = "notes/WORKBOARD.md"
INTERVIEW_PATH = "notes/INTERVIEW.md"

FIGURE_EXT = (".svg", ".png", ".jpg", ".jpeg", ".pdf")

# Confidence tiers. `measured` is worth ten times a guess, which is roughly the
# ratio at which the four claims falsified on 2026-08-30 should have been
# separated from the ones that survived.
TIER_WEIGHT = {"measured": 1.0, "derived": 0.7, "inferred": 0.3, "guess": 0.1}

# Thirty independent units is full weight. That is not a statistical claim, it
# is the scale this project works at: the largest paired sample so far is 22.
SAMPLE_SATURATION = 30

# One dataset is capped at a half. This is the red card as arithmetic: no amount
# of work on a single corpus can carry a claim past 0.5 on its own.
INDEPENDENCE = {0: 0.0, 1: 0.5, 2: 0.8}
INDEPENDENCE_MAX = 1.0

# Unpaired comparisons are discounted because SPEC V38 measured that raw planner
# error is dominated by how non-linear a clip happens to be.
PAIRED_WEIGHT = {True: 1.0, False: 0.6}

# What each wording tier costs. Rising through them requires more evidence and,
# above `scoped`, the user's explicit sign-off.
TIER_BAR = [("universal", 0.8), ("comparative", 0.6), ("existential", 0.5),
            ("scoped", 0.3)]
TIER_ORDER = ["scoped", "existential", "comparative", "universal"]

CONTESTED_THRESHOLD = 0.3


class NoWeights(Exception):
    """A criteria set has no weights, so it cannot be scored yet."""


# --------------------------------------------------------------------------
# evidence
# --------------------------------------------------------------------------

def w_n(n):
    """Sample weight, with diminishing returns and no false precision."""
    if not n or n <= 0:
        return 0.0
    return min(1.0, math.log10(1 + n) / math.log10(1 + SAMPLE_SATURATION))


def independence(datasets):
    """How much a claim's corpus coverage is worth.

    Counts **distinct** datasets. A dataset is a published data collection with
    a name and an author or organisation; categories and splits within one
    corpus count once.
    """
    return INDEPENDENCE.get(len(set(datasets or [])), INDEPENDENCE_MAX)


def tier_of(observation):
    """The confidence tier, derived from provenance rather than chosen.

    A stored `tier` is honoured only when it *lowers* the value. Claiming
    `measured` with no source is not believed, which removes most of the
    discretion over the largest multiplier.
    """
    stated = observation.get("tier")
    earned = "measured" if observation.get("source") else "inferred"
    if stated and TIER_WEIGHT.get(stated, 1.0) < TIER_WEIGHT[earned]:
        return stated
    return earned


def evidence_strength(observation):
    """`e` in [0, 1] for one observation. Never stored."""
    return (TIER_WEIGHT[tier_of(observation)]
            * w_n(observation.get("n"))
            * independence(observation.get("datasets"))
            * PAIRED_WEIGHT[bool(observation.get("paired"))])


def _by_dataset(observations):
    """The strongest observation per dataset.

    Taking the maximum **within** a dataset is what stops five runs on one
    corpus from looking like five independent findings.
    """
    best = {}
    for o in observations:
        for ds in (o.get("datasets") or ["?"]):
            e = evidence_strength(o)
            if e > best.get(ds, -1):
                best[ds] = e
    return best


def _noisy_or(values):
    out = 1.0
    for v in values:
        out *= (1.0 - v)
    return 1.0 - out


def support(observations):
    """Combined support: max within a dataset, combined across datasets."""
    yes = [o for o in observations if o.get("supports", True)]
    return _noisy_or(_by_dataset(yes).values())


def oppose(observations):
    no = [o for o in observations if not o.get("supports", True)]
    return _noisy_or(_by_dataset(no).values())


def contested(observations, threshold=CONTESTED_THRESHOLD):
    """Real evidence on both sides. The flag worth a human look."""
    return min(support(observations), oppose(observations)) >= threshold


# --------------------------------------------------------------------------
# claims
# --------------------------------------------------------------------------

def licensed_tier(e):
    """The strongest wording this much evidence permits, or None."""
    for name, bar in TIER_BAR:
        if e >= bar:
            return name
    return None


def wording_tier(claim, e):
    """What a claim may actually say: the lower of asked-for and licensed.

    A claim is never promoted above what it asked for, and it is silently
    demoted when the evidence does not reach the bar.
    """
    licensed = licensed_tier(e)
    if licensed is None:
        return None
    asked = claim.get("claim_type", "scoped")
    if asked not in TIER_ORDER:
        return licensed
    return min([asked, licensed], key=TIER_ORDER.index)


# --------------------------------------------------------------------------
# decisions
# --------------------------------------------------------------------------

def decide(scores, weights):
    """Weighted sum. Raises when the criteria set has no weights yet."""
    if not weights:
        raise NoWeights("no weights set for this kind of decision")
    return sum(weights.get(k, 0.0) * v for k, v in scores.items())


def sensitivity(candidates, weights):
    """The smallest single weight change that would flip the top two.

    Reported with every ranking. A decision this fragile is reported rather
    than taken.
    """
    ranked = sorted(((decide(s, weights), name)
                     for name, s in candidates.items()), reverse=True)
    if len(ranked) < 2:
        return None
    (top_score, top), (second_score, second) = ranked[0], ranked[1]
    gap = top_score - second_score
    best = None
    for crit in weights:
        delta = candidates[top][crit] - candidates[second][crit]
        if abs(delta) < 1e-9:
            continue
        needed = gap / delta                      # weight change that closes it
        target = weights[crit] - needed
        if target < 0 or target > 1:
            continue
        move = abs(target - weights[crit])
        if best is None or move < best["move"]:
            best = {"criterion": crit, "from": weights[crit], "to": round(target, 3),
                    "move": move, "top": top, "second": second, "gap": gap}
    return best


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def load(path=None):
    path = path or PLAN_PATH
    if not os.path.isfile(path):
        return {"units": [], "assumptions": [], "claims": [], "questions": [],
                "availability": [], "observations": [], "decisions": {}}
    with open(path) as handle:
        return json.load(handle)


def check(plan):
    """Everything wrong with the plan, as a list of sentences."""
    problems = []
    unit_ids = set(u["id"] for u in plan.get("units", []))
    assumption_ids = set(a["id"] for a in plan.get("assumptions", []))
    observation_ids = set(o["id"] for o in plan.get("observations", []))

    for u in plan.get("units", []):
        uid = u.get("id", "?")
        for aid in u.get("assumes", []):
            if aid not in assumption_ids:
                problems.append("%s assumes %s, which does not exist" % (uid, aid))
        for did in u.get("depends_on", []):
            if did not in unit_ids:
                problems.append("%s depends on %s, which does not exist" % (uid, did))
        if u.get("state") == "accepted" and not u.get("accepted_by"):
            problems.append("%s is accepted with no record of who accepted it; "
                            "only the author may accept" % uid)
        if u.get("state") in ("evidence_produced", "accepted"):
            evidence = u.get("evidence") or []
            if not evidence:
                problems.append("%s claims evidence was produced but lists none"
                                % uid)
            for path in evidence:
                if not os.path.exists(path):
                    problems.append("%s lists evidence %s, which is not on disk"
                                    % (uid, path))
            if u.get("produces") == "measurement" and not any(
                    p.lower().endswith(FIGURE_EXT) for p in evidence):
                problems.append("%s produces a measurement and no figure; every "
                                "result needs something to look at" % uid)

    for c in plan.get("claims", []):
        for oid in c.get("evidence", []):
            if oid not in observation_ids:
                problems.append("claim %s cites observation %s, which does not "
                                "exist" % (c.get("id", "?"), oid))
        for aid in c.get("rests_on", []):
            if aid not in assumption_ids:
                problems.append("claim %s rests on %s, which does not exist"
                                % (c.get("id", "?"), aid))

    for o in plan.get("observations", []):
        src = o.get("source")
        if src and not os.path.exists(src):
            problems.append("observation %s cites %s, which is not on disk"
                            % (o.get("id", "?"), src))
    return problems


def contradictions(plan):
    """Falsified assumptions still holding units up, and contested claims."""
    out = []
    falsified = set(a["id"] for a in plan.get("assumptions", [])
                    if a.get("falsified"))
    for u in plan.get("units", []):
        if u.get("state") == "accepted":
            for aid in u.get("assumes", []):
                if aid in falsified:
                    out.append("%s is accepted but rests on %s, which was "
                               "falsified" % (u["id"], aid))
    by_id = {o["id"]: o for o in plan.get("observations", [])}
    for c in plan.get("claims", []):
        obs = [by_id[i] for i in c.get("evidence", []) if i in by_id]
        if obs and contested(obs):
            out.append("claim %s is contested: support %.2f against %.2f"
                       % (c.get("id", "?"), support(obs), oppose(obs)))
    return out


def _ready(plan, unit, runs_on=None):
    if unit.get("state") not in (None, "not_started", "in_progress"):
        return False
    if unit.get("blocked_on"):
        return False
    if runs_on and unit.get("runs_on") != runs_on:
        return False
    done = set(u["id"] for u in plan.get("units", [])
               if u.get("state") in ("evidence_produced", "accepted"))
    return all(d in done for d in unit.get("depends_on", []))


def next_unit(plan, runs_on=None):
    """The single unit worth starting now, or None."""
    ready = [u for u in plan.get("units", []) if _ready(plan, u, runs_on)]
    if not ready:
        return None
    phase_order = {p: i for i, p in enumerate(
        ["P0", "P1", "P2", "P3", "P4", "P6", "P7", "T1", "T2"])}
    return sorted(ready, key=lambda u: (phase_order.get(u.get("phase"), 99),
                                        u.get("cost_days", 1.0),
                                        u.get("id", "")))[0]


# --------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------

def render_board(plan):
    lines = ["# Work board", "",
             "*Generated by `tools/workplan.py render`. Do not edit by hand;"
             " edit `WORKPLAN.json` or `NOW.md`.*", ""]
    units = plan.get("units", [])
    if not units:
        lines.append("No units yet.")
    by_phase = {}
    for u in units:
        by_phase.setdefault(u.get("phase", "?"), []).append(u)
    for phase in sorted(by_phase):
        lines += ["## %s" % phase, "",
                  "| id | title | state | runs on | next action |",
                  "|---|---|---|---|---|"]
        for u in sorted(by_phase[phase], key=lambda x: x.get("id", "")):
            state = u.get("state", "not_started")
            if u.get("blocked_on"):
                state = "blocked: %s" % u["blocked_on"].get("who", "?")
            lines.append("| `%s` | %s | %s | %s | %s |"
                         % (u.get("id"), u.get("title", ""), state,
                            u.get("runs_on", ""), u.get("next_action", "")))
        lines.append("")
    waiting = [u for u in units if u.get("state") == "evidence_produced"]
    if waiting:
        lines += ["## Waiting for you to accept", ""]
        for u in waiting:
            lines.append("- `%s` — %s" % (u.get("id"), u.get("title", "")))
        lines.append("")
    return "\n".join(lines) + "\n"


def render_interview(plan):
    lines = ["# Interview", "",
             "*Generated. Questions first, then things to check.*", ""]
    open_q = [q for q in plan.get("questions", []) if not q.get("answered")]
    lines += ["## Questions", ""] if open_q else ["## Questions", "",
                                                  "None open.", ""]
    for q in open_q:
        lines += ["### %s — %s" % (q.get("id"), q.get("asks", "")),
                  "",
                  "**Why it matters.** %s" % q.get("why_it_matters", ""),
                  "",
                  "**Until you answer** I will: %s"
                  % q.get("default_if_unanswered", "wait"), ""]
    pending = [a for a in plan.get("availability", [])
               if a.get("status") in (None, "unknown", "user_checking")]
    lines += ["## Things to check", ""]
    if not pending:
        lines.append("Nothing pending.")
    for a in pending:
        lines += ["### %s — %s" % (a.get("id"), a.get("what", "")), "",
                  a.get("directions", ""), "",
                  "*Needed for:* %s" % ", ".join(a.get("needed_for", [])), ""]
    return "\n".join(lines) + "\n"


def render_score(plan):
    by_id = {o["id"]: o for o in plan.get("observations", [])}
    lines = ["claim  e     tier         wording      inputs"]
    for c in plan.get("claims", []):
        obs = [by_id[i] for i in c.get("evidence", []) if i in by_id]
        e = max([evidence_strength(o) for o in obs] or [0.0])
        best = max(obs, key=evidence_strength) if obs else None
        detail = ""
        if best:
            detail = ("tier=%s n=%s datasets=%d paired=%s"
                      % (tier_of(best), best.get("n"),
                         len(set(best.get("datasets") or [])),
                         bool(best.get("paired"))))
        lines.append("%-6s %.2f  %-12s %-12s %s"
                     % (c.get("id"), e, c.get("claim_type", "-"),
                        wording_tier(c, e) or "none", detail))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["next", "render", "check", "score",
                                        "contradictions", "report"])
    ap.add_argument("--runs-on", default=None)
    ap.add_argument("--plan", default=None)
    a = ap.parse_args(argv)
    plan = load(a.plan)

    if a.command == "check":
        problems = check(plan)
        if not problems:
            print("plan is consistent: %d units, %d assumptions, %d claims"
                  % (len(plan.get("units", [])), len(plan.get("assumptions", [])),
                     len(plan.get("claims", []))))
            return 0
        print("%d problem(s):\n" % len(problems))
        for p in problems:
            print("  %s" % p)
        return 1

    if a.command == "contradictions":
        out = contradictions(plan)
        for c in out:
            print("  %s" % c)
        print("%d contradiction(s)" % len(out))
        return 1 if out else 0

    if a.command == "next":
        unit = next_unit(plan, a.runs_on)
        if unit is None:
            print("nothing actionable")
            return 0
        print("%s  %s" % (unit.get("id"), unit.get("title", "")))
        print("  why:  %s" % unit.get("why", ""))
        print("  next: %s" % unit.get("next_action", ""))
        print("  done when: %s" % unit.get("done_when", ""))
        return 0

    if a.command == "render":
        for path, text in ((BOARD_PATH, render_board(plan)),
                           (INTERVIEW_PATH, render_interview(plan))):
            directory = os.path.dirname(path)
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            with open(path, "w") as handle:
                handle.write(text)
            print("wrote %s" % path)
        return 0

    if a.command == "score":
        print(render_score(plan))
        return 0

    if a.command == "report":
        by_rubric = {}
        for u in plan.get("units", []):
            by_rubric.setdefault(u.get("rubric", "-"), []).append(u)
        for r in sorted(by_rubric):
            print("\n## Rubric %s" % r)
            for u in by_rubric[r]:
                print("  - [%s] %s" % (u.get("state", "?"), u.get("title", "")))
                for ev in u.get("evidence", []):
                    print("      evidence: %s" % ev)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
