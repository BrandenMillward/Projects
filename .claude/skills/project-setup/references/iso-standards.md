# ISO Standards

Deciding whether any formal standard applies to this project, and if so, wiring it in so the evidence falls out of work you were doing anyway.

## Start here: most projects need none of this

For a solo build, a side project, a tool for one person, or a portfolio piece, the correct answer is **no ISO standard applies**, and `docs/STANDARDS.md` should say so in one line. Adopting a standard nobody asked for produces documents nobody reads and a false impression of rigour.

Ask one question, and take "no" for an answer:

> Does anything about this project's context require a formal standard — a customer contract, a regulated domain, a certification the organisation already holds, or a procurement process you'll have to pass?

If the answer is no, write "No formal standards regime applies" and move on. If it's "maybe later", note the trigger condition instead of adopting it now.

## Two things not to conflate

**Alignment** is structuring your artifacts so they satisfy the intent of a clause. It is free, it is useful, and it is what these skills can do.

**Certification** is an accredited body auditing a management system — organisational scope, records, internal audits, management review. No repository, and no skill, produces it.

Never write "ISO compliant" in a generated document. Write "aligned with", name the clause, and let the reader judge. Overstating this is the fastest way to make every other claim in the doc set suspect.

## The map

Where a standard does apply, these are the ones that touch how a software project is planned, specified, built, and changed. Each row says what it actually asks of you and where that lands in the doc set.

| Standard | Governs | What it asks for | Lands in |
|---|---|---|---|
| **ISO/IEC/IEEE 12207** | Software life cycle processes | Defined processes with declared inputs and outputs across the life cycle | The skills themselves — phases, gates, records |
| **ISO/IEC/IEEE 29148:2018** | Requirements engineering | Requirements that are necessary, implementation-free, unambiguous, consistent, complete, singular, feasible, traceable and verifiable | `docs/project_spec.md` §1 — the specificity standard is already this |
| **ISO/IEC 25010:2023** | Product quality model | Named quality characteristics with stated targets | `docs/STANDARDS.md` quality attributes; spec §3 engineering requirements |
| **ISO/IEC 5055:2021** | Automated source code quality | Measures for security, reliability, performance efficiency and maintainability, taken from source | The linter, type checker and CI suite |
| **ISO/IEC 27001:2022** | Information security management | 93 Annex A controls across four themes (organizational, people, physical, technological). A.8.32 covers change management: planned, assessed, authorised, tested, documented | `.env` policy, secret-scanning hook, change records |
| **ISO 9001:2015** | Quality management | §8.5.6 control of changes — retain documented information on the review result, who authorised it, and any actions arising | Change records (`project-change`) |
| **ISO/IEC 20000-1:2018** | IT service management | Change control for services in operation. *Clause number unverified — check the standard before citing one* | Change records plus deployment notes |

### The one that pays for itself regardless

**ISO/IEC 25010:2023's quality model** is worth borrowing even when no standard applies, because it is a checklist against forgetting a whole dimension. Its nine characteristics: functional suitability, performance efficiency, compatibility, interaction capability, reliability, security, maintainability, flexibility, and safety.

Walk them once during engineering design. Most will be "not a concern for this project" — say so explicitly. The two or three that matter get a number, and those numbers become the engineering requirements that were going to be vague otherwise. "It should be fast" fails the specificity bar; "p50 under 10s, measured" is a quality attribute.

## Regulated domains

If the project ships into one, the domain standard governs and the general set above is the floor:

- **Medical device software** — IEC 62304, plus ISO 14971 for risk management.
- **Automotive** — ISO 26262 functional safety.
- **AI management systems** — ISO/IEC 42001:2023.

Say plainly that these are out of scope for what this skill covers, and that the project needs domain-specific guidance rather than a generic scaffold. Pretending otherwise in a regulated context is worse than useless.

## Writing it up

Fill the "Applicable standards" table in `docs/STANDARDS.md` with one row per adopted standard: the standard, why it applies, the clause or characteristic being aligned with, and where the evidence lives. An adopted standard with no evidence column is a claim, not an alignment.

Then stop. The point of doing this at setup is that later changes inherit it automatically — `project-change` reads `docs/STANDARDS.md` and gates against whatever is declared there. If nothing is declared, its ISO gate is a no-op, which is the correct behaviour for almost every project.
