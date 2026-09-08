# Change Intake

Turning "can we make it do X" into something that can be assessed, decided, and later verified.

## The bar

Same standard the spec was written to, pointed at a request instead of a requirement: **a change request only counts if a finished implementation could fail it.**

| Fails the bar | Passes the bar |
|---|---|
| Add search | Users filter entries by keyword and date range from the entry list; results show the matched phrase highlighted, ranked by recency. Replaces the current unfiltered list. |
| Make it faster | Thumbnail generation drops from ~25s to under 10s p50, with visible progress state throughout. No change to output quality. |
| Move to Postgres | Replace MongoDB with Supabase Postgres. Existing entries migrate with IDs preserved. Read paths change; the write API does not. |
| Support teams | Out of scope — see the not-in-scope row. Revisit when a second paying user asks. |

Three things separate the columns, and all three are things the assessment will need anyway:

1. **What changes** — the new behaviour, stated as observable outcome.
2. **What it replaces** — the old behaviour, explicitly. A request that only says what to add and never says what it displaces is how features accumulate.
3. **What stays the same** — the boundary. This is what makes "did we break something" answerable later.

When the request comes in vague, **write the specific version yourself and ask them to correct it.** "I'm reading that as: … Right?" Same technique as the interview, same reason: asking someone to be more specific returns the work to them and usually produces another vague sentence.

## Classify it

One line in the record, and it drives which gates apply later:

| Type | Typical trigger | Usually earns |
|---|---|---|
| `feature` | new capability | spec amendment, milestone row |
| `change` | existing behaviour altered | spec amendment, migration note |
| `deprecation` | capability removed | callers audit, removal notice |
| `migration` | same behaviour, different substrate | data plan, rollback plan |
| `dependency` | new or upgraded third-party | licence, cost, supply-chain check |
| `security` | vulnerability or hardening | emergency lane, security gate |
| `fix` | restore intended behaviour | usually no record — check the threshold |

## The three exits

**Approve** — it goes to impact assessment. Nothing is approved *at* intake; intake only decides that the request is well-formed enough to assess.

**Defer** — right idea, wrong time. Write it into the not-in-scope row of `docs/project_spec.md` with two things attached: **why not now**, and **the condition that would change the answer**. A deferral without a revisit condition is a rejection nobody wants to admit to, and it will be re-litigated every month until someone does.

**Reject** — it conflicts with the purpose, the constraints, or a decision already recorded in `docs/DECISIONS.md`. Say which, in one sentence, and record it. A rejection with a reason attached is the only thing that stops the same request arriving again in a different costume.

## The emergency lane

For a production break or a security fix that cannot wait:

1. **Fix it.** Do not open a record first.
2. Record it within the same session, marked `retroactive: true`, with what happened, what was changed, and what was *not* verified in the rush.
3. Add one line: **what would have caught this earlier** — a test, a hook, a monitor, a rejected change that should have been approved.
4. If the rushed fix left debt, that debt is a new request at Phase 1. Name it now while it is still embarrassing; in a week it will be invisible.

The retroactive flag matters. A change register where every entry looks like it followed the process is a register being written after the fact, and it will not survive contact with anyone who checks.

## Requirements characteristics worth borrowing

ISO/IEC/IEEE 29148 defines what makes a requirement well-formed. Four of its characteristics apply to change requests directly, because they are the ones that fail in practice:

- **Singular** — one change per request. If the request contains "and", check whether it is two.
- **Unambiguous** — one reading only. If two people would build different things from the sentence, it is not ready.
- **Verifiable** — there is an observation that settles whether it worked.
- **Traceable** — it links back to a requirement, a job to be done, or a recorded decision. A change that traces to nothing is a change nobody asked for.

Do not turn this into a checklist ritual in front of the user. Apply it silently while writing the request, and raise only the one that actually fails.
