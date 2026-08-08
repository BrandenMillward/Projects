---
description: Turn a milestone from the project spec into GitHub issues, one per capability
allowed-tools: Bash(gh*), Read, Glob
---

Create GitHub issues for a milestone.

Target milestone: $ARGUMENTS (e.g. `MVP`, `v1`). If not given, use the current milestone from
`docs/PROJECT_STATUS.md`.

1. Read `docs/project_spec.md` — the milestone table and the requirements it maps to.
2. Read `docs/ARCHITECTURE.md` for the component names, so issues can name where the work lands.
3. Check existing issues first (`gh issue list --state all`). Don't create duplicates; if
   something is already tracked, say so and skip it.
4. Draft one issue per capability in the milestone. Show the full list to the user for
   approval **before** creating anything — creating issues is outward-facing and tedious to
   undo. Each issue:

   ```
   Title: <imperative capability, e.g. "User can download the generated thumbnail">

   ## What
   <the capability, at the spec's level of specificity — entry point, mechanism, outcome>

   ## Why
   <the job to be done or requirement from docs/project_spec.md this serves>

   ## Done when
   - [ ] <observable, checkable condition>
   - [ ] <...>

   ## Notes
   <components involved, known gotchas, dependencies on other issues>
   ```

5. Split anything that isn't completable in one sitting. An issue that takes a week is a
   milestone wearing a costume.
6. After approval, create them with a milestone label if one exists, and report the URLs.
