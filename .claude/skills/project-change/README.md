# project-change

A Claude Code skill that governs change to a project that already exists. Sibling to
`project-setup`: setup draws the baseline, this is how the baseline moves.

**The prime directive:** the spec is amended before the code, in the same change record.
A codebase that has drifted from its spec is one where the spec has stopped being the
source of truth — and after that, no later document can be trusted either.

**Five phases, one gate:** intake at the specificity bar → six-surface impact assessment
→ decision → execute (spec first) → verify and close out. Plus a Phase 0 threshold check,
because most work is a commit and should never open a record at all.

## Install

Install globally, alongside `project-setup`:

```bash
mkdir -p ~/.claude/skills
cp -r .claude/skills/project-change ~/.claude/skills/
```

## Use

Describe the change. It triggers on change language without needing to be named:

- "add keyword search to the entry list"
- "we need to move off Mongo"
- "is dropping the v1 endpoint a breaking change?"
- `/change`, `/impact`, `/close-change`, `/standards-check`

To skip the process for something small, say so — the threshold rule is explicit and the
skill will tell you when it thinks a record isn't warranted.

## The threshold

A record is opened only when the change alters a requirement, crosses a milestone
boundary, adds a dependency or paid service, changes an interface others depend on,
touches anything in `docs/STANDARDS.md`, or is rated a one-way door. Everything else is
a commit. There is also an emergency lane: fix first, record after, marked `retroactive`.

## Standards

Two gates, both reading the project's own `docs/STANDARDS.md` (written by `project-setup`):

- **Coding standards** — format, lint, types, tests, dependency audit, secret scan as
  automated gates, then a judgement pass over the diff for the things linters miss.
- **ISO standards** — only where a project has actually declared one. Most declare none,
  and the skill says so rather than inventing ceremony. Where one applies, the change
  record *is* the evidence: ISO 9001:2015 §8.5.6 wants the review result, the authoriser,
  and the actions arising; ISO/IEC 27001:2022 A.8.32 wants changes planned, assessed,
  authorised, tested and documented. Both fall out of the record's own fields.

Nothing here produces certification, and the skill never writes "compliant" — the accurate
word is "aligned with".

## Layout

```
SKILL.md                          orchestrator: phases, the threshold, the gate
references/
  change-intake.md                the bar for requests, defer/reject, emergency lane
  impact-analysis.md              six surfaces, what counts as breaking
  reversibility.md                cheap / annoying / one-way door, and how to downgrade
  standards-gates.md              code gate + ISO clause-to-evidence map
  verification-and-closeout.md    evidence by change type, the close-out sequence
assets/
  templates/                      change-record, CHANGE_REGISTER, DECISIONS
  commands/                       change, impact, close-change, standards-check
  agents/                         impact-analyzer, spec-drift, standards-auditor
  hooks/settings.example.json     hard gates for secrets and lint; advisory drift warning
scripts/changes.py                IDs, records, register, validation
```

## changes.py

The skill runs this for you, but it works standalone:

```bash
python3 scripts/changes.py new "Replace MongoDB with Supabase" --type migration
python3 scripts/changes.py set CHG-0001 --status approved --reversibility one-way-door \
  --decided-by branden
python3 scripts/changes.py validate CHG-0001
python3 scripts/changes.py set CHG-0001 --status shipped
python3 scripts/changes.py register
```

Records live in `docs/changes/CHG-NNNN-slug.md`; `register` rebuilds `docs/CHANGE_REGISTER.md`
from them, so the index is never state anyone maintains by hand.

`validate` scales with status: an open record only needs its request written, a shipped one
needs the impact, decision, reversibility rating, a named authoriser, and at least one ticked
verification item. `--dry-run` works on every subcommand.

## Customising it

- `references/change-intake.md` — the threshold rule and the classification table.
- `references/standards-gates.md` — the toolchain table and which ISO standards you care about.
- `assets/templates/change-record.md` — the fields. Anything you add here shows up on every
  future record, so add sparingly.
