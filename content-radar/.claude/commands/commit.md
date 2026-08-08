---
description: Stage changes and commit with a conventional-commit message derived from the diff
allowed-tools: Bash(git status*), Bash(git diff*), Bash(git add*), Bash(git commit*), Bash(git log*), Read
---

Commit the current work.

1. Run `git status` and `git diff` (and `git diff --staged`) to see what actually changed. Read the diff — the message must describe the change, not restate the file names.
2. Run `git log --oneline -10` to match the repository's existing message style.
3. Group the changes. If the diff contains two unrelated changes, say so and propose splitting into separate commits rather than writing a vague message that covers both.
4. Stage the relevant files. Never stage `.env`, credentials, or build output — if any are present, stop and flag it.
5. Commit with a conventional-commit message:

```
<type>(<scope>): <imperative summary under 72 chars>

<why the change was made, if it isn't obvious from the summary>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `style`.

$ARGUMENTS may contain a hint about the intent of the change — use it to inform the message, but let the diff be the authority.

Do not push. Do not open a PR.
