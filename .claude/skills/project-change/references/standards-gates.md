# Standards Gates

Two checks every change passes through, both driven by the project's own `docs/STANDARDS.md`.

If that file doesn't exist, the code gate still applies (every project has a formatter and a linter, or should) and the ISO section is a no-op. Propose creating `docs/STANDARDS.md` from `project-setup`'s template, but never block a change on its absence.

## Gate 1 — Coding standards

Runs on every change that touches code. Machine-checkable first: **a standard a human has to remember is a standard that erodes.**

### The automated pass

Run what the project actually has, in this order, and stop at the first hard failure:

| Check | Python | TypeScript | Swift |
|---|---|---|---|
| Format | `ruff format --check` | `prettier --check` | `swift-format lint` |
| Lint | `ruff check` | `eslint` | `swiftlint` |
| Types | `mypy` | `tsc --noEmit` | compiler |
| Tests | `pytest` | `vitest` / `jest` | `swift test` |
| Deps | `pip-audit` | `npm audit` | — |
| Secrets | staged-diff scan for key patterns and `.env` | same | same |

Format, lint, and the secret scan are **hard gates** — they fail the change. Types and tests are hard gates once the project has them meaningfully. Dependency audit is advisory unless the finding is high severity in a runtime dependency.

Report failures as: the command, the failing file and line, and the fix. Never report "lint failed" without the finding.

### The judgement pass

Things no linter catches. Apply to the diff only, not the whole codebase:

- **Error handling** — no swallowed exceptions, no bare catch hiding a bug, failures surface with enough context to debug.
- **Naming** — matches the surrounding code's conventions, not a new dialect introduced by this change.
- **Function and file size** — a change pushing a function past readable length is the moment to split it, not later.
- **Dead surface** — code the change makes unreachable is deleted in the same change.
- **Comments explain why.** A comment restating the code is noise; a comment explaining a non-obvious decision is the most valuable line in the file.
- **Tests encode behaviour, not implementation.** A test that breaks on every refactor is a liability.
- **Security of the diff** — injection surfaces, authorization checks on new endpoints, secrets never logged, user input never trusted. Cross-check against the OWASP Top 10 categories relevant to what changed rather than the whole list.

Dispatch `standards-auditor` for this pass when the diff is large enough that reading it inline would crowd the session.

## Gate 2 — ISO standards

**Read this first: most projects need none of this, and saying so is the correct answer.** Only run this gate when `docs/STANDARDS.md` names a standard as applicable.

Alignment with a clause is not certification. Nothing this skill produces should ever be described as "ISO compliant" — the accurate phrasing is "aligned with", and the distinction matters to anyone who is ever audited. Certification requires an accredited body assessing a management system, not a tidy repo.

Where a standard does apply, the goal is that the evidence falls out of work already being done. The change record is the evidence.

### What a change record satisfies

| Standard | What it asks of a change | Where it lands |
|---|---|---|
| **ISO 9001:2015 §8.5.6** — control of changes | Retain documented information on the **results of the review**, the **person authorising**, and **any actions arising** | `impact`, `decided_by`, `follow_ups` |
| **ISO/IEC 27001:2022 A.8.32** — change management | Changes planned, assessed for impact, authorised, tested, documented | The five phases; `verification` section |
| **ISO/IEC/IEEE 12207** — software life cycle processes | A defined process with declared inputs and outputs | The record is the declared output |
| **ISO/IEC/IEEE 29148:2018** — requirements engineering | Requirements singular, unambiguous, verifiable, traceable | Intake bar; `traces_to` |
| **ISO/IEC 25010:2023** — product quality model | Named quality characteristics with targets | Quality attributes in `docs/STANDARDS.md` |
| **ISO/IEC 5055:2021** — automated source code quality | Security, reliability, performance efficiency, maintainability, measured in source | Gate 1's automated pass |
| **ISO/IEC 20000-1:2018** — service management | Change control for services in operation. *Clause number not verified — check the standard before citing one.* | Same record, plus a deployment note |

### Quality-attribute impact

ISO/IEC 25010:2023 names nine characteristics: functional suitability, performance efficiency, compatibility, interaction capability, reliability, security, maintainability, flexibility, and safety.

When a change touches one that has a stated target in `docs/STANDARDS.md`, measure it and record the number. A performance target with no measurement attached to the change that moved it is a target nobody is holding.

### Regulated domains

If the project ships into one — medical (IEC 62304), automotive (ISO 26262), or an AI management system (ISO/IEC 42001) — these gates are the floor, not the ceiling, and the domain standard's own change-control requirements govern. Say that plainly rather than implying this skill covers it.

## Recording the result

Three lines in the change record, no more:

```
Code gate      pass — ruff, tsc, 34 tests green; 2 advisory dep findings noted
Standards      n/a — no ISO regime declared for this project
Quality        p50 generation 8.4s (target <10s) — measured, moved the right way
```

A gate result of "pass" with no evidence behind it is the failure mode. Name the commands that ran.
