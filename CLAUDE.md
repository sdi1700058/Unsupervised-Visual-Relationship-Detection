# sgg-thesis

FOSAE thesis. Hard deadline **26 October 2026**.

Two repositories, nested. `sgg-thesis` is public and holds code. `workbench`
inside it is private and holds notes, plans, the queue and the runs. Never
run `git clean -x` here: it destroys the nested repository.

## Start here, every session

Read these, in this order. It is section 1 of `notes/docs/DEV.md`, copied
here because a pointer gets skipped and this does not. `notes/` is a symlink
to `workbench/notes`, so every path resolves from this directory.

1. `notes/OVERNIGHT_PROMPT.md` — how the work is done. The authority on
   working rules.
2. `notes/NOW.md` — the queue. The author writes it by hand and it outranks
   the task list in item 1.
3. `notes/PROGRESS.md` — where the work stands, and the open problems.
4. `notes/docs/SUPERVISOR.md` — what the thesis is marked against.
5. `notes/docs/STE.md` — the writing style. Before every write.
6. `notes/docs/DEV.md` — the full index: what each of the 29 documents
   governs, and when to read it. Go here whenever you do not know which
   document rules a question.

```bash
python3 workbench/tools/workplan.py next     # what to work on
python3 workbench/tools/workplan.py check    # is the plan consistent
```

On 2026-09-14 a session worked nine hours, opened five of the 29 documents,
never opened the index and never ran the gate below. Both were reachable and
neither was reached, so both are inlined or probed now rather than named.

| File | What it governs |
|---|---|
| `notes/docs/DEV.md` | **the index. Read order and what each file is for** |
| `notes/docs/WORKING_RULES.md` | style, process, code, the claim ladder |
| `notes/docs/STE.md` | the writing rules, before every write |
| `notes/docs/AGENTIC_PROCEDURE.md` | how to run the loop, step by step |
| `notes/docs/AGENTIC_DESIGN.md` | why the loop is shaped this way |
| `notes/docs/SANDBOX.md` | the confinement, and what permission covers |

## The gate answers most of it for you

```bash
bash workbench/sh/gate.sh
```

Ten checks, one exit code. It reads the wiki's own rules: whether a
document cites a file that is not there, whether any Markdown gained a
writing violation, whether a withdrawn number reads as current, whether the
plan is consistent. Run it before you commit anything that touches the
notes.

Two of its checks are also probe rows, `documents` and `writing_rules`, so
every batch answers them whether or not anyone runs the gate.

## Rules that fail silently, so they are here and not only there

**Never state a number you did not measure.** Run the command, then write
the number. On 2026-09-14 an estimate of "9 units instead of 126" went into
a plan without a command behind it. The measured answer was 48. Every claim
carries its strength: `measured`, `derived`, `inferred` or `guess`, and
anything below `measured` says so in the sentence that carries it.

**Nine identical measurements are a finding.** One is not. Runs 23 to 30
each reported zero acceptances and each was read as a fresh defect rather
than as the ninth point on a flat line.

**Run the checker on every Markdown you write**:
`python3 workbench/tools/check_ste.py --paths <file>`. It reports how many
violations a file gained, so a file that rose is a file you broke.

**Never `git add -A`.** Name the paths. On 2026-09-13 it committed a hand's
stray file twice, and both had to be untracked afterwards.

**Confirm every file deletion with the author, one at a time**, even when a
plan already approved it.

**Never push.** What is published is the author's decision alone.

**Nothing is written outside `/scratch/users/konkotsa/panos`.** The Sherlock
account belongs to the author's brother.

**Python packages go in the venv**, never global and never user-site.

## Before committing anything under `workbench/tools/`

```bash
cd workbench && PYTHONPATH=.. ../.venv-local/bin/python -m unittest discover -s tools/planner/tests
python3 workbench/tools/check_py36.py --root .
python3 workbench/tools/probe.py --rows workbench/tools/probes.json
```

The suite count is asserted by the `private_suite` probe row, so a new test
means updating that row in `workbench/tools/probes.json` with the reason.

## Sherlock

`/etc/claude-code/CLAUDE.md` governs the cluster: modules, Slurm, storage.
It is not repeated here. The login node is shared, so heavy work goes to a
job.
