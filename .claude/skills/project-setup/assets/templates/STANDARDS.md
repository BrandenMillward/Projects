# {{PROJECT_NAME}} — Standards

> Last updated: {{DATE}}
> What "correct" means in this project: how the code is written, how it's checked, and
> which formal standards (if any) apply. `project-change` reads this file to decide
> which gates a change must clear.

---

## 1. Coding standards

### Rules

<!-- Keep this short. Anything a linter can enforce belongs in the config, not here.
     Delete the ones that don't apply — a list nobody reads past rule six is worthless. -->

- Naming is descriptive and consistent with the file it lives in.
- A function that needs a comment to explain its sections is two functions.
- Never swallow an exception. Failures surface with enough context to debug from the message.
- Comments explain *why*. A comment restating the code is noise.
- Every dependency is a permanent liability — added with a reason, not a reflex.
- Secrets never appear in code, logs, or a commit.
- Tests encode behaviour, not implementation.
- Dead code is deleted, not commented out.

### Toolchain

| Check | Command | Gate |
|---|---|---|
| Format | {{FORMAT_CMD}} | hard — blocks |
| Lint | {{LINT_CMD}} | hard — blocks |
| Types | {{TYPES_CMD}} | hard once meaningful |
| Test | {{TEST_CMD}} | hard once meaningful |
| Dependency audit | {{AUDIT_CMD}} | advisory unless high severity in a runtime dep |
| Secret scan | staged-diff scan for key patterns and `.env` | hard — blocks |

**Strictness:** <!-- "warnings are errors in CI" | "linter stays out of the way" -->

**Test expectation:** <!-- everything | core loop only | none yet (and why) -->

### Enforcement

1. Editor — format on save
2. Hook — format and lint on write; block staged secrets
3. CI — full suite on push
4. Review — only what the first three cannot check

---

## 2. Quality attributes

Derived from the ISO/IEC 25010:2023 quality model. Most rows will be "not a concern" —
say so explicitly rather than deleting the row, so the next person knows it was considered
rather than forgotten. Every row that *is* a concern needs a number.

| Characteristic | Target for this project |
|---|---|
| Functional suitability | |
| Performance efficiency | |
| Compatibility | |
| Interaction capability | |
| Reliability | |
| Security | |
| Maintainability | |
| Flexibility | |
| Safety | |

---

## 3. Applicable standards

<!-- For most projects the honest answer is the default line below. Adopting a standard
     nobody asked for produces documents nobody reads and a false impression of rigour.
     Delete the default line only if a standard genuinely applies. -->

**No formal standards regime applies to this project.**

<!-- If one does, replace the line above with rows. "Aligned with" is the accurate phrase —
     never "compliant". Certification requires an accredited body auditing a management
     system, which no repository produces. -->

| Standard | Why it applies | Clause aligned with | Evidence lives in |
|---|---|---|---|
| | | | |

**Revisit when:** <!-- the trigger that would change the answer — a customer contract,
                      a regulated deployment, a procurement process -->
