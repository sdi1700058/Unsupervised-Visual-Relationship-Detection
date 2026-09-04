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
        asked = licensed
    tier = min([asked, licensed], key=TIER_ORDER.index)
    # Evidence licenses a tier; only the author grants it. Without this, enough
    # evidence would silently promote a sentence in a document nobody had read.
    if tier != "scoped" and not claim.get("approved_by"):
        return "scoped"
    return tier


# --------------------------------------------------------------------------
# claim wording, generated into the documents
# --------------------------------------------------------------------------

CLAIM_START = "<!-- claim: %s -->"
CLAIM_END = "<!-- /claim -->"

# How each tier is allowed to speak. The scoped form deliberately reports an
# event and draws no consequence; that is the whole point of it.
# The author's own instruction on how a claim should read: "we did this
# experiment E on these specific samples S and had this result R, so it is
# possible". So every run is listed on its own line with its own sample and its
# own result, and the consequence the tier permits comes last and separately.
# An earlier version printed only the strongest observation, which hid the
# other corpus entirely while still counting it towards the strength.
TIER_CONSEQUENCE = {
    "scoped": "That is what those runs produced. Nothing follows from it "
              "beyond them.",
    "existential": "So it is possible on that data. Nothing is claimed about "
                   "other data, other clips, or the method in general.",
    "comparative": "So the difference holds across the corpora tested, and not "
                   "necessarily beyond them.",
    "universal": "The evidence spans enough corpora to state this generally.",
}

TIER_LABEL = {"scoped": "Scoped claim", "existential": "Existential claim",
              "comparative": "Comparative claim", "universal": "Universal claim"}


def experiment_of(observation):
    """A readable name for the run an observation came from.

    Taken from the source path, because that is the thing a reader can open.
    An `experiment` field overrides it when the path is not self-explaining.
    """
    named = observation.get("experiment")
    if named:
        return named
    source = observation.get("source") or ""
    parts = [p for p in source.split("/") if p]
    for part in reversed(parts):
        if part not in ("summary.csv", "metrics.json") and "." not in part:
            return part
    return parts[-2] if len(parts) > 1 else (source or "an unrecorded run")


def claim_evidence(claim, plan):
    """The observations a claim rests on."""
    by_id = dict((o["id"], o) for o in plan.get("observations", []))
    return [by_id[i] for i in claim.get("evidence", []) if i in by_id]


def claim_strength(claim, plan):
    """A claim's evidence strength, combined **across** corpora.

    Not the strongest single observation. Scoring a claim by its best
    observation left the independence multiplier inescapable: a second dataset
    changed nothing, so the arithmetic expression of "one corpus is not enough"
    could never be satisfied by doing the obvious thing. Registering VidOR
    beside VidVRD and watching C1 still read `datasets=1` is how that surfaced.

    `support` takes the maximum within a dataset and combines across datasets,
    so repeating a run on one corpus still buys nothing.
    """
    obs = claim_evidence(claim, plan)
    return support(obs) if obs else 0.0


def claim_sentence(claim, plan):
    """The sentence a claim is licensed to make. Derived, never stored.

    The tier decides the frame and the strongest observation supplies the
    facts, so the wording cannot outrun the evidence: to say more, add an
    observation.
    """
    obs = claim_evidence(claim, plan)
    if not obs:
        return ("**%s has no evidence.** Nothing may be written from it."
                % claim.get("id", "This claim"))
    best = max(obs, key=evidence_strength)
    e = claim_strength(claim, plan)
    tier = wording_tier(claim, e)
    if tier is None:
        return ("**%s: evidence inconclusive** (strength %.2f, below the %.2f "
                "floor). Current hypothesis, not a finding: %s."
                % (claim.get("id"), e, TIER_BAR[-1][1], claim.get("asserts")))
    datasets = sorted(set(d for o in obs for d in (o.get("datasets") or [])))
    lines = ["**%s.**" % TIER_LABEL[tier], ""]
    for o in sorted(obs, key=lambda x: experiment_of(x)):
        corpora = ", ".join(o.get("datasets") or ["an unnamed corpus"])
        lines.append("- In **%s**, on %s clips of %s: %s."
                     % (experiment_of(o), o.get("n"), corpora,
                        (o.get("what") or "").rstrip(".")))
        if o.get("caveat"):
            lines.append("  *%s*" % o["caveat"])
    lines += ["",
              "%s Evidence strength **%.2f** across **%d dataset%s**."
              % (TIER_CONSEQUENCE[tier], e, len(datasets),
                 "" if len(datasets) == 1 else "s")]
    return "\n".join(lines)


def render_claims(plan):
    """Write each claim's licensed sentence into the documents that state it."""
    written = []
    for claim in plan.get("claims", []):
        body = claim_sentence(claim, plan)
        start = CLAIM_START % claim["id"]
        for path in claim.get("appears_in", []):
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            if start not in text or CLAIM_END not in text.split(start, 1)[1]:
                continue
            head, rest = text.split(start, 1)
            _, tail = rest.split(CLAIM_END, 1)
            fresh = head + start + "\n\n" + body + "\n\n" + CLAIM_END + tail
            if fresh != text:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(fresh)
                written.append(path)
    return written


def check_claims(plan):
    """Documents whose stated wording has drifted from what is licensed."""
    problems = []
    for claim in plan.get("claims", []):
        cid = claim.get("id", "?")
        appears = claim.get("appears_in") or []
        e = claim_strength(claim, plan)
        licensed = licensed_tier(e)
        if licensed and licensed != "scoped" and not claim.get("approved_by"):
            asked = claim.get("claim_type", "scoped")
            grantable = (min([asked, licensed], key=TIER_ORDER.index)
                         if asked in TIER_ORDER else licensed)
            if grantable != "scoped":
                problems.append(
                    "%s awaits your sign-off: at strength %.2f the evidence "
                    "licenses '%s', and it stays written as scoped until you "
                    "grant it" % (cid, e, grantable))
        if not appears:
            problems.append("%s names no document; a claim nobody states is a "
                            "claim nobody can check" % cid)
            continue
        body = claim_sentence(claim, plan)
        start = CLAIM_START % cid
        for path in appears:
            if not os.path.isfile(path):
                problems.append("%s appears_in %s, which is not on disk"
                                % (cid, path))
                continue
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            if start not in text or CLAIM_END not in text.split(start, 1)[1]:
                problems.append("%s has no generated block in %s; add the "
                                "markers so the wording cannot drift"
                                % (cid, path))
                continue
            current = text.split(start, 1)[1].split(CLAIM_END, 1)[0].strip()
            if current != body.strip():
                problems.append("%s is out of date in %s; run "
                                "tools/workplan.py render" % (cid, path))
    return problems


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
# milestones, counted from what is on disk
# --------------------------------------------------------------------------

def combinations_measured(plan=None, root="."):
    """(dataset, method) pairs whose declared evidence is on disk.

    Declared in the plan, not detected by globbing. Globbing matched
    `eval/probe/vidor` and `eval/probe/se_batch`, which hold oracle exports
    rather than probe results, and missed two real results at the same time.
    A milestone counter has to be the harder of the two to fool, so it reads a
    declaration and then checks the file is there.
    """
    plan = load() if plan is None else plan
    out = []
    for combo in plan.get("combinations", []):
        path = os.path.join(root, combo.get("evidence", ""))
        if combo.get("evidence") and os.path.exists(path):
            out.append("%s x %s" % (combo["dataset"], combo["method"]))
    return sorted(out)


def papers_fully_treated(index_path="notes/lit/index.json"):
    """Papers carrying all three of summary, deeper notes, and eval detail.

    Two of the three is not one paper. `notes/lit/deep/<id>.md` holds the
    deeper notes; nothing writes it yet, so this reads zero until that work
    starts, which is the honest answer.
    """
    if not os.path.isfile(index_path):
        return []
    with open(index_path) as handle:
        index = json.load(handle)
    out = []
    for paper in index.get("papers", []):
        has_summary = bool(paper.get("data_text")) and paper.get("has_mechanism")
        has_detail = bool(paper.get("datasets")) or bool(paper.get("metrics"))
        has_notes = os.path.isfile("notes/lit/deep/%s.md" % paper["id"])
        if has_summary and has_detail and has_notes:
            out.append(paper["id"])
    return out


def milestone_progress(plan):
    """Countable progress per milestone. Counts artifacts, not intentions."""
    counters = {
        "combinations_measured": lambda: combinations_measured(plan),
        "papers_fully_treated": lambda: papers_fully_treated(),
        "parked": lambda: [],
    }
    rows = []
    for m in plan.get("milestones", []):
        got = counters.get(m.get("counter"), lambda: [])()
        units = [u for u in plan.get("units", [])
                 if u.get("milestone") == m["id"]]
        rows.append({
            "id": m["id"], "title": m["title"],
            "have": len(got), "target": m.get("target", 0),
            "detail": got,
            "units_total": len(units),
            "units_accepted": sum(1 for u in units
                                  if u.get("state") == "accepted"),
        })
    return rows


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
        e = claim_strength(c, plan)
        detail = ""
        if obs:
            datasets = set(d for o in obs for d in (o.get("datasets") or []))
            detail = ("observations=%d n=%d datasets=%d"
                      % (len(obs), sum(o.get("n") or 0 for o in obs),
                         len(datasets)))
        lines.append("%-6s %.2f  %-12s %-12s %s"
                     % (c.get("id"), e, c.get("claim_type", "-"),
                        wording_tier(c, e) or "none", detail))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["next", "render", "check", "score",
                                        "contradictions", "report",
                                        "sensitivity", "graph", "claims",
                                        "progress"])
    ap.add_argument("--runs-on", default=None)
    ap.add_argument("--plan", default=None)
    a = ap.parse_args(argv)
    plan = load(a.plan)

    if a.command == "progress":
        rows = milestone_progress(plan)
        print("%-5s %-52s %-11s %s" % ("", "milestone", "counted", "units"))
        for r in rows:
            bar = "%d of %d" % (r["have"], r["target"])
            print("%-5s %-52s %-11s %d accepted of %d"
                  % (r["id"], r["title"][:52], bar,
                     r["units_accepted"], r["units_total"]))
        for r in rows:
            if r["detail"]:
                print("\n%s counts:" % r["id"])
                for item in r["detail"]:
                    print("  %s" % (item,))
        return 0

    if a.command == "claims":
        for c in plan.get("claims", []):
            print("%s  %s" % (c["id"], claim_sentence(c, plan)))
            print()
        return 0

    if a.command == "sensitivity":
        for kind, spec in sorted(plan.get("decisions", {}).items()):
            weights = spec.get("weights")
            print("%s: %s" % (kind, "no weights set" if not weights
                              else "weights %s" % weights))
        print("\nPass candidate scores to workplan.sensitivity() to see what "
              "would flip a ranking; DATASETS.md carries the current one.")
        return 0

    if a.command == "graph":
        print("dataset and metric usage is charted by "
              "tools/lit/render_index.py, which writes "
              "notes/lit/dataset_usage.svg")
        return 0

    if a.command == "check":
        problems = check(plan) + check_claims(plan)
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
        for touched in render_claims(plan):
            print("wrote the claim block in %s" % touched)
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
