# Verification and Close-out

The half of change management that gets skipped, and the half that makes the register worth keeping.

**Unverified is not shipped.** A change record marked done because the code was written is a record that says nothing. The question at close-out is the one the request answered at intake: *how would you know it worked?*

## Verification by change type

Match the evidence to what was claimed. One or two lines each — this is proof, not a test report.

| Type | What counts as evidence |
|---|---|
| `feature` | The new behaviour exercised end to end, by the entry point named in the request. A test that covers it. |
| `change` | Old behaviour gone, new behaviour present, and the explicitly unchanged parts still unchanged. |
| `deprecation` | No remaining callers (show the search), and the removal notice where consumers will see it. |
| `migration` | Row counts before and after, a spot-check of migrated records, and the rollback path exercised at least once. |
| `dependency` | Build and tests green on the new version, audit clean, licence checked. |
| `security` | The vulnerable path is closed — demonstrated, not asserted — and a regression test exists. |
| `performance` | The measurement, against the target in `docs/STANDARDS.md`. A claim without a number is not evidence. |

If a verification criterion from the decision could not be met, the record says so and the change stays `open`. Shipping with a known unmet criterion is a decision the user makes explicitly, recorded as such — not a silent downgrade.

## The close-out sequence

Run in this order; each step depends on the last.

1. **Verify** against the criteria in the record. Paste the evidence in.
2. **Amend the spec** if execution diverged from the plan. This happens more often than anyone admits, and the record is where the divergence is caught.
3. **`docs/CHANGELOG.md`** — one entry, Keep a Changelog shape (Added / Changed / Fixed / Removed), written for someone skimming for "when did that break?". Describe the user-visible or developer-visible change, not the diff. Reference the change ID.
4. **`docs/PROJECT_STATUS.md`** — update the milestone row's status, add to what's been accomplished with the date, and re-cut "what's next" if this change reordered it. Keep next to three to five items.
5. **`docs/DECISIONS.md`** — only when a real decision was made: a fork in the road where the alternative was viable. Context, decision, consequences, alternatives rejected. Not every change earns one; a change that had no alternative had no decision.
6. **Mark the record `shipped`** with the date. `changes.py set <id> --status shipped` and `changes.py register` to rebuild the index.

## When it goes wrong

**Mark it `reverted`, don't delete it.** Add what happened, what the assessment missed, and what would have caught it. A register in which nothing ever failed is a register nobody is being honest with, and the failed entries are the ones with anything to teach.

Then ask the compounding question: *was this a bad change, or a gap in the process that assessed it?* If the impact assessment missed a surface, that surface belongs in `references/impact-analysis.md`. If a hook would have caught it, propose the hook. This is where the skill improves itself — the same mechanism the `retro` agent provides for sessions.

## Post-change review

Only for changes rated one-way door, or any change that got reverted. Three questions, a few lines each:

- Did the impact assessment match what actually happened?
- Was the reversibility rating right?
- What would you want to know before doing the next change like this one?

Everything else closes without ceremony. A review process applied to routine changes is how the whole system becomes something people work around.
