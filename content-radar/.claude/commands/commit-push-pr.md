---
description: Commit, push to the current branch, and open a pull request
allowed-tools: Bash(git*), Bash(gh*), Read
---

Commit the current work, push it, and open a PR.

1. Commit following the same rules as `/commit` — read the diff, match existing style, refuse to stage secrets.
2. Check the branch. If on the default branch, create a feature branch first (`git checkout -b <type>/<short-slug>`) — don't commit straight to main.
3. Push with `git push -u origin <branch>`. On network failure, retry up to four times with backoff (2s, 4s, 8s, 16s).
4. Look for a PR template (`.github/pull_request_template.md`, `.github/PULL_REQUEST_TEMPLATE.md`, root, or `docs/`). If one exists, use its headings as the layout and fill them in from the actual changes. Skip any section asking for credentials or internal hostnames.
5. Open the PR. Title matches the commit summary. Body covers:
   - **What changed** — the substance, not a file list
   - **Why** — the requirement or milestone from `docs/project_spec.md` this serves
   - **How to verify** — the commands or steps a reviewer runs
6. Report the PR URL.

$ARGUMENTS may contain the PR title or extra context for the body.
