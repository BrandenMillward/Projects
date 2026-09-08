# Impact Analysis

Establishing what a change actually touches, before anything is touched.

The output is the middle section of the change record and the input to the decision. It is not a survey of the codebase — it is an answer to one question: **what breaks, and what has to change with it?**

## Run it in a forked context

Dispatch the `impact-analyzer` subagent. This is the textbook case for one: the work requires reading widely across the repo, and none of that reading should end up in the session that then does the implementation. Only the report survives, which is exactly the property you want.

Do it inline only when the change is obviously contained to one file you already have open.

## The six surfaces

Work through these in order. Stop early when a surface comes back empty — an assessment listing five "no impact" sections is padding.

### 1. Spec

The most important surface and the one most often skipped. Which requirements does this change make untrue, incomplete, or newly ambiguous? Which milestone row does it belong in? Does it pull something out of "not in scope for now" — and if so, is that a decision the user has actually made?

Quote the existing requirement text and write the amended version. Do not describe the amendment in the abstract; the diff is the deliverable.

### 2. Code

- **Direct**: the components implementing the behaviour being changed.
- **Callers**: everything depending on the interfaces being changed. `Grep` for the symbol, don't reason about it from memory.
- **Tests**: which tests encode the old behaviour and therefore must change. A change requiring no test change is either trivial or untested — decide which, and say so.
- **Dead surface**: what this change makes unreachable. Removal is part of the change, not follow-up work.

### 3. Data

The surface with the least reversible failures.

- Schema change, and whether it is additive (safe) or destructive (not).
- Migration: forward path, and the backfill for existing rows.
- **Rollback**: can the migration be reversed with the data intact? If not, this change is a one-way door regardless of how small the diff is.
- Retention and personal data: does this change what is stored about a person, or for how long?

### 4. Config and cost

New environment variables (and their entries in `.env.example`), new external services, new permissions or scopes, new per-operation cost. If the project has a cost ceiling in its engineering requirements, check the change against it and state the number.

### 5. Standards

Which gates the change must clear — see `references/standards-gates.md`. Most changes clear the code-quality gate and nothing else. Flag explicitly when a change touches an area a standard in `docs/STANDARDS.md` governs, because that is where the evidence obligations attach.

### 6. Docs

Which documents go stale the moment this ships: `ARCHITECTURE.md` when a component's responsibility moves, `docs/reference/<feature>.md` when behaviour changes, `CLAUDE.md` when a constraint or command changes. These are part of the change, not cleanup after it.

## What counts as a breaking change

Be precise, because "breaking" is the word deciding whether a change needs a migration path, a version bump, and a notice.

**Breaking:**

- an interface changes shape or meaning for an existing caller — signature, response body, error contract, status codes
- previously valid input is rejected, or previously rejected input is silently accepted
- a default changes in a way that alters existing behaviour without action
- data written by the old version cannot be read by the new one, or vice versa
- a capability is removed, renamed, or moved

**Not breaking:**

- adding an optional field, parameter, or endpoint
- internal refactor with identical observable behaviour
- performance change within the stated bar
- adding a new capability behind a flag that defaults off

The test that settles arguments: **could something that worked yesterday stop working today without anyone changing it?** If yes, it is breaking. For a solo project with a single caller this may still be fine — "breaking" is a fact about the change, not a verdict on it.

## Writing the assessment

Keep it to one screen. The reader is deciding, not studying.

```
IMPACT — CHG-0007  Replace MongoDB with Supabase Postgres

Spec        §3 tech stack (database row); §1 requirement 4 unaffected
Code        3 components, 11 call sites in data/; entry read path rewritten
Data        Full migration, ~4k rows. IDs preserved. Reverse migration untested.
Config      +SUPABASE_URL, +SUPABASE_ANON_KEY; -MONGODB_URI. Cost +$0 at MVP scale.
Standards   Code gate only. No ISO regime applies to this project.
Docs        ARCHITECTURE.md system overview; reference/storage.md rewritten

Reversibility  ONE-WAY DOOR — reverse migration untested and data is the asset
Recommendation Approve, but write and test the reverse migration first
```

Close with a recommendation and the reason. An assessment that lays out facts and refuses to recommend has left the hardest part of the work undone.
