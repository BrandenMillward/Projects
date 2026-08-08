---
description: Update the project docs to match the code, then commit code and docs together
allowed-tools: Bash(git*), Read, Edit, Write, Glob, Grep
---

Bring the documentation in line with what changed, then commit everything in one commit.

Docs updated in the same commit as the code are the only docs that stay true — that's the
whole point of this command. Docs updated "later" are docs updated never.

## 1. Read the change

`git status`, `git diff`, `git diff --staged`. Understand what actually changed and, more
importantly, whether it changed anything a reader of the docs would now be wrong about.

## 2. Update each doc — only where the change warrants it

Be selective. Touching every doc on every commit produces noise that trains everyone to
ignore doc diffs.

- **`docs/CHANGELOG.md`** — almost always. Add an entry under `[Unreleased]` in the right
  group (Added / Changed / Fixed / Removed). Describe the change as a reader experiences it.
- **`docs/PROJECT_STATUS.md`** — if this completed something. Move it into "Accomplished"
  with today's date, update the milestone status, and trim "What's next".
- **`docs/ARCHITECTURE.md`** — only if a component was added, removed, or changed
  responsibility, or the request flow changed. Not for ordinary edits inside a component.
- **`docs/reference/<feature>.md`** — if a documented feature's behaviour, interface, or
  config changed. If the change *created* a feature complex enough that someone would
  otherwise read the code to understand it, create the reference doc now.
- **`CLAUDE.md`** — if a command, convention, constraint, or the architecture summary is now
  wrong. Also delete anything that has stopped being true. Deleting is the maintenance work;
  keep it under ~100 lines.
- **`docs/project_spec.md`** — only if the *product* changed, not the implementation. If it
  did, say so explicitly in the commit message, because a silently drifting spec is how a
  project ends up building something nobody agreed to.

If a doc genuinely needs no change, skip it and say which you skipped and why.

## 3. Commit

Stage the code and the doc updates together. Never stage `.env`, credentials, or build
output — if any appear, stop and flag it.

```
<type>(<scope>): <imperative summary under 72 chars>

<why, if not obvious>

Docs: <which docs were updated>
```

Do not push unless asked. $ARGUMENTS may describe the intent of the change.
