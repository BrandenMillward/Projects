# Coding Standards

What "good code" means in this project, decided once at setup so it never has to be argued per pull request.

The output is `docs/STANDARDS.md` and a working toolchain. Both matter: a standards document with no tooling behind it is a document that describes what the code used to look like.

## The principle

**Machine-checkable beats memorable.** Every rule you can push into a formatter, a linter, or a type checker is a rule that costs nothing to follow and never erodes. Every rule that lives only in prose is one that decays at the exact rate people get busy.

So the order of preference is always: make it automatic → make it a hook → make it a CI check → write it down. Only the things that genuinely need judgement should end up as prose.

## The house baseline

Language-agnostic, and short enough that people actually read it. These go into `docs/STANDARDS.md` and get referenced from `CLAUDE.md`:

- **Naming** — descriptive over clever, and consistent with the file it lives in. New dialects introduced mid-codebase cost more than the naming they improve.
- **Function size** — if it needs a comment to explain its sections, it is two functions.
- **Error handling** — never swallow an exception. Failures surface with enough context to debug from the message alone.
- **Comments explain why.** A comment restating the code is noise; a comment explaining a non-obvious decision is the highest-value line in the file.
- **Dependencies** — each one is a permanent liability. Adding one is a decision with a reason, not a reflex. Check licence, maintenance, and transitive weight.
- **Secrets** — never in code, never in logs, never in a commit. `.env` is gitignored before the first commit, and checked again before the first push.
- **Tests** encode behaviour, not implementation. A test that breaks on every refactor is a liability disguised as coverage.
- **Dead code is deleted**, not commented out. Git remembers.

Resist making this list longer. A twenty-rule standard is one nobody has read past rule six.

## Toolchain by stack

Pick from the project's chosen stack, install the config at setup, and wire the hooks. These are the current defaults — check versions rather than assuming.

| | Python | TypeScript / Node | Swift |
|---|---|---|---|
| Format | `ruff format` | `prettier` | `swift-format` |
| Lint | `ruff check` | `eslint` | `swiftlint` |
| Types | `mypy` (or `pyright`) | `tsc --noEmit`, `strict: true` | compiler |
| Test | `pytest` | `vitest` or `jest` | `XCTest` |
| Security | `bandit`, `pip-audit` | `npm audit` | — |
| Config lands in | `pyproject.toml` | `eslint.config.js`, `tsconfig.json`, `.prettierrc` | `.swiftlint.yml` |

`ruff` replaces the old flake8/isort/black stack and is fast enough to run on every write, which is what makes it worth choosing. For TypeScript, `strict: true` from day one — retrofitting strictness onto a codebase that grew without it is a project in itself.

## Security baseline

Not ISO, but the checks that actually prevent incidents. Scope them to what the project does rather than adopting the whole catalogue:

- **OWASP Top 10** — the reference for anything with a web surface. Check the categories relevant to what a change touches, not the full list every time.
- **CWE Top 25** — useful when reviewing a specific class of bug.
- **Dependency audit** in CI, and on every dependency change.
- **Secret scanning** on the staged diff, as a blocking hook. This is the single highest-value gate in a solo project, because a leaked key is the most likely expensive mistake.

## Conventions worth fixing early

- **Conventional Commits** — `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `style`. Already what `/commit` writes. It makes changelog generation mechanical.
- **Semantic versioning** once anything else consumes the project. Before that it is ceremony.
- **Branch naming** — one pattern, written in `CLAUDE.md`, so branches sort usefully.

## The enforcement ladder

State this explicitly when presenting the standard, because it is what makes the difference between a standard and a wish:

1. **Editor** — format on save. Zero friction, catches most of it.
2. **Hook** — `PostToolUse` formats and lints what was just written; `PreToolUse` blocks staged secrets. The harness runs these, so they are rules rather than suggestions.
3. **CI** — the full suite on every push. Catches what ran only on someone else's machine.
4. **Review** — the judgement pass. Reserved for what the first three genuinely cannot check.

Anything enforced at level 4 that could be enforced at level 1 is wasting a person's attention.

## What to ask the user

Two questions, batched, with recommendations first:

1. **Strictness** — is this a personal project where the linter should stay out of the way, or one where CI should fail on a warning? This single answer determines the lint config more than any individual rule.
2. **Test expectation** — tests for everything, tests for the core loop only, or none yet? "None yet" is a legitimate answer for an MVP, but it should be a decision rather than a drift, and it belongs in `docs/STANDARDS.md` so the next session knows.

Then write `docs/STANDARDS.md` from the template, install the configs, and wire the hooks in the same pass.
