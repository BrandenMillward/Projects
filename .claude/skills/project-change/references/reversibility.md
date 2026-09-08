# Reversibility

One field on every change record, and the field that decides how much process the change earns.

This generalises the line that already closes the stack research report in `project-setup`: *the one decision that would be most expensive to reverse later*. That question is worth asking of every change, not just the stack.

## The three tiers

### Cheap

Undone by reverting the commit. No data written in a new shape, no external state created, no interface anyone else has consumed.

**Earns:** the record, and nothing else. Do the work. Approval can be implicit if the request was approved.

### Annoying

Reversible, but not by `git revert` alone. Data has been migrated, a service provisioned, a config propagated, a cache warmed with the new shape.

**Earns:** a written rollback path in the record — the actual steps, not "restore from backup" — and a check that the rollback has been *tried at least once* if the change touches data. An untested rollback plan is a hypothesis.

### One-way door

Cannot be undone in any practical sense. Data destroyed or transformed lossily, a public interface others depend on removed, a vendor commitment made, a key rotated, anything published outward.

**Earns:** an explicit second look before execution, a named person deciding, and the alternatives considered written into `docs/DECISIONS.md`. This is the only tier where slowing down is the point rather than the cost.

## How to rate it

Ask in this order; the first "yes" sets the tier:

1. Does this destroy or lossily transform data? → one-way door
2. Does it remove or change something outside the project's control — a published API, a sent message, a rotated credential, a deleted account? → one-way door
3. Does undoing it require steps beyond reverting code? → annoying
4. Otherwise → cheap

**Size is not reversibility.** A 2,000-line refactor with no interface change is cheap. A one-line change to a retention policy that starts deleting user data is a one-way door. Rate the consequence, never the diff.

## Reducing the tier

Often the best outcome of an assessment is not "approve" or "reject" but **"approve a cheaper version of this."** Standard moves, worth proposing before accepting a one-way door:

- **Expand then contract.** Add the new shape alongside the old, migrate readers, remove the old later as a separate change. Turns one irreversible change into two reversible ones.
- **Flag it.** Ship dark, enable for yourself, enable broadly later. The rollback becomes a config toggle.
- **Copy, don't move.** Write to the new store while still writing the old, until the new one has proven itself.
- **Soft delete.** Mark rows dead, purge them in a separate change once the window has passed.
- **Test the reverse first.** For migrations, write and run the down-path before running the up-path. This alone converts most one-way doors into merely annoying ones.

If none of these apply and the change is genuinely irreversible, say so plainly in the record. "This cannot be undone" is a fact the decision needs, and softening it is a disservice.
