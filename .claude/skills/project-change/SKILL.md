---
name: project-change
description: Run a controlled change against a project that already exists — intake the request at the specificity bar, assess blast radius and reversibility, get a decision, amend the spec before the code, check the work against the project's coding standards and any ISO standards that apply, then verify and close it out with a change record. Use whenever someone wants to add, alter, remove, or migrate something in a project that already has a spec or a codebase: "add a feature", "change the stack", "we need to support X", "deprecate Y", "migrate to Z", "is this a breaking change", "what would this affect", a hotfix that needs recording after the fact, or a request to review the impact or standards conformance of a change before doing it. Also use for /change, /impact, /close-change, and /standards-check.
---

# Project Change

Govern a change to a project that already exists. `project-setup` draws the baseline; this skill is how the baseline moves.

The difference is the starting state, and it changes everything. Setup drafts a spec from nothing. Here the spec is a **baseline you diff against**, and the question is never "what should this be?" — it is **"what does this change break, and is that worth it?"**

## The prime directive

**The spec is amended before the code, in the same change record.** Every other rule here exists to serve that one. A codebase that has drifted from its spec is a codebase where the spec has stopped being the source of truth, and once that happens no later document can be trusted either.

## Phase 0 — Does this need a change record at all?

Ask this first, every time. Most work does not need a record, and a process that ceremonializes every edit gets routed around inside two weeks.

**Open a record when the change does any of these:**

- alters, adds, or removes a requirement in `docs/project_spec.md`
- crosses a milestone boundary, or moves something out of "not in scope for now"
- adds a dependency, a paid service, or a new data flow
- changes an interface other code or other people depend on
- touches auth, secrets, personal data, or anything named in `docs/STANDARDS.md`
- is rated **one-way door** (see `references/reversibility.md`)

**Otherwise: just do the work and commit.** A bug fix inside existing behaviour, a refactor with no interface change, a copy edit, a dependency patch bump — these are commits, not changes. Say which you've decided and why in one line, then proceed. Do not ask the user to adjudicate every edit.

**The emergency lane.** Production is broken, or a security fix cannot wait. Fix it now, record it after, mark the record `retroactive: true`, and note in the record what would have caught it earlier. A change process with no emergency lane teaches everyone that the process is optional — this is the pressure valve that keeps the rest of it honest.

## Phase 1 — Intake

One sentence, at the same specificity bar the spec was written to: **a change request only counts if a finished implementation could fail it.** Read `references/change-intake.md` — it has the bar, the rewrite technique, and the three ways a request can leave this phase.

A request can end here. `defer` and `reject` are real outcomes, and writing "not now, because X, revisit when Y" into the not-in-scope row is a successful run of this skill.

## Phase 2 — Impact assessment

Establish blast radius before touching anything. Read `references/impact-analysis.md`, then dispatch the `impact-analyzer` subagent — this is exactly the work that should happen in a forked context, because it reads broadly and only the report should survive.

Assess in this order, and stop early if the answer is "no":

1. **Spec** — which requirements, jobs, and milestones does this touch?
2. **Code** — which components, interfaces, and callers?
3. **Data** — schema, migration, backfill, and whether it is reversible.
4. **Config** — new env vars, services, permissions, costs.
5. **Standards** — which gates in `docs/STANDARDS.md` this change must clear. See `references/standards-gates.md`.
6. **Docs** — which of the doc set goes stale the moment this ships.

Then rate reversibility — **cheap / annoying / one-way door** — using `references/reversibility.md`. That rating is what decides how much process the rest of this change earns. A cheap, reversible change gets a light touch; a one-way door earns the full treatment and an explicit second look.

## Phase 3 — Decision

**Gate:** show the assessment and ask for a decision. Approve, defer, reject, or reduce scope. Record who decided and when — that field is not bureaucracy, it is the single thing most standards actually ask for.

If the change is approved with conditions ("yes, but behind a flag"), the conditions go in the record as verification criteria, not as a verbal aside.

## Phase 4 — Execute

In this order, without exception:

1. **Amend the spec.** Requirement text, milestone row, engineering design — whatever this change makes untrue.
2. **Do the work**, keeping to the change record's scope. Anything discovered mid-flight that is out of scope becomes a new request at Phase 1, not a quiet addition.
3. **Run the standards gates** for the areas the assessment flagged — `references/standards-gates.md` covers both the code-quality gate and the ISO clause evidence. `/standards-check` runs them.

## Phase 5 — Verify and close

Read `references/verification-and-closeout.md`. In short: prove the change did what the request said, then update `CHANGELOG.md`, `PROJECT_STATUS.md`, `DECISIONS.md` where a real decision was made, and mark the record `shipped`. Unverified is not shipped.

A change that turned out to be wrong gets a `reverted` record with what was learned, not a deleted one. The register is a history, and the failures are the valuable entries.

## Standards

Two gates, both project-specific, both read from the project's own `docs/STANDARDS.md`:

- **Coding standards** — the house code rules and the toolchain that enforces them. Machine-checkable wherever possible, because a standard a human has to remember is a standard that erodes.
- **ISO standards** — only the ones the project actually needs. Most solo projects need none, and saying so is the correct answer. Where one does apply, the job is to make the evidence fall out of work you were doing anyway.

Both live in `references/standards-gates.md`. Read it at Phase 2, not upfront.

If the project has no `docs/STANDARDS.md`, propose creating one from `project-setup`'s template rather than inventing a standard per change — but do not block the change on it.

## When there is no spec

If the project has no `docs/project_spec.md`, there is no baseline to diff against and this skill has nothing to stand on. Say so, and offer `project-setup`'s recovery mode: read the code, draft the spec from what is actually there, get it corrected, and *then* run the change. Do not silently invent a baseline — a fabricated spec is worse than none, because everything after it inherits the fabrication.

## Running this without being exhausting

The failure mode of a change-management skill is ceremony. Guard against it the same way `project-setup` guards against interrogation:

- **The threshold rule is load-bearing.** Use it. Most work is a commit.
- **Scale process to reversibility, not to size.** A large, cheaply reversible change needs less process than a small one-way door.
- **One record, one change.** If the assessment finds two unrelated changes, split them.
- **Batch the questions.** Impact assessment and the decision are one `AskUserQuestion` call, not six.
- **Never re-ask what the record already answers.** The record is the state; read it before asking anything.

## Files in this skill

- `references/change-intake.md` — the specificity bar for requests, the defer/reject paths, the emergency lane
- `references/impact-analysis.md` — blast radius method, what counts as a breaking change
- `references/reversibility.md` — the three tiers and what process each earns
- `references/standards-gates.md` — coding-standard gate and the ISO clause-to-evidence map
- `references/verification-and-closeout.md` — evidence, doc updates, post-change review
- `assets/templates/` — change record, register, decision record
- `assets/commands/`, `assets/agents/`, `assets/hooks/` — commands, subagents, enforcement hooks
- `scripts/changes.py` — allocates IDs, writes records, maintains the register, validates
