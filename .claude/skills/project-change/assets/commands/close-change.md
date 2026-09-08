---
description: Verify a change, update the doc set, and mark the record shipped
allowed-tools: Bash(python3*), Bash(git*), Bash(npm*), Bash(pytest*), Bash(ruff*), Read, Grep, Glob, Edit, Write
---

Close out change: $ARGUMENTS

**Unverified is not shipped.** Work in this order — each step depends on the last.

1. **Verify** against the criteria in the record, matched to the change type (see `references/verification-and-closeout.md`). Paste real evidence: the command that ran, the number measured, the search showing no remaining callers. A claim without evidence fails this step.

   If a criterion could not be met, say so and leave the record open. Shipping with a known unmet criterion is the user's explicit call, recorded as such.

2. **Amend the spec** if execution diverged from the plan. This is where divergence gets caught.

3. **`docs/CHANGELOG.md`** — one entry in Keep a Changelog shape, describing the visible change rather than the diff, referencing the change ID.

4. **`docs/PROJECT_STATUS.md`** — milestone status, what's been accomplished with the date, and re-cut "what's next" if this reordered it. Keep next to three to five items.

5. **`docs/DECISIONS.md`** — only if a real decision was made, with a viable alternative. Most changes don't earn one.

6. **Mark it shipped and rebuild the register:**

```bash
python3 <skill-dir>/scripts/changes.py set <id> --status shipped --dir .
python3 <skill-dir>/scripts/changes.py register --dir .
```

If the change went wrong, mark it `reverted` instead — never delete it — and add what the assessment missed and what would have caught it.

Do not commit. Leave that to `/commit` or `/update-docs-and-commit`.
