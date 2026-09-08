---
description: Run the project's coding-standard gate, and the ISO gate if any standard applies
allowed-tools: Bash, Read, Grep, Glob, Task
---

Run the standards gates over: $ARGUMENTS (default: the current diff against the base branch)

**Gate 1 — coding standards.** Read `docs/STANDARDS.md` for the project's declared toolchain; fall back to detecting it from the repo if that file doesn't exist.

Run format, lint, types, tests, dependency audit, and a secret scan of the staged diff — in that order, stopping at the first hard failure. Format, lint, and the secret scan are hard gates. Report every failure as: the command, the file and line, and the fix. Never report "lint failed" without the finding.

Then the judgement pass over the diff only — error handling, naming consistency, function size, dead surface, comments that explain why, tests that encode behaviour rather than implementation, and the security of what changed. Dispatch `standards-auditor` when the diff is large enough that reading it inline would crowd the session.

**Gate 2 — ISO standards.** Only if `docs/STANDARDS.md` names a standard as applicable. Most projects declare none, and reporting "no ISO regime applies" is a correct and complete result.

Where one does apply, check the clause-to-evidence map in `references/standards-gates.md` and report which obligations this change satisfies and which are outstanding. Never describe anything as "ISO compliant" — the accurate word is "aligned with".

**Report** in the three-line form:

```
Code gate      <pass/fail — commands that ran>
Standards      <n/a, or clause and evidence>
Quality        <measurement against target>
```

Fix nothing unless asked. This command reports.
