# Documentation System

The docs exist so Claude Code can be effective in this repo without re-deriving the project every session, and so the user can tell what's true months later. Both goals fail the same way: documents that grow until nobody reads them.

The structure below is a hub and spokes. `CLAUDE.md` is the hub — always in context, deliberately short, mostly pointers. Everything else is a spoke, read on demand.

```
CLAUDE.md                    <- always in context; keep under ~100 lines
.env / .env.example
docs/
├── project_spec.md          <- requirements, milestones, engineering design
├── ARCHITECTURE.md          <- system overview + component architecture
├── CHANGELOG.md             <- what changed, newest first
├── PROJECT_STATUS.md        <- milestones, done, next
└── reference/
    └── <feature>.md         <- one per key feature
```

## CLAUDE.md

This is project memory. It's in context for every single turn, so every line has to earn its place — bloat here is a tax on every request the user ever makes, and a CLAUDE.md nobody trusts is worse than none.

Include, briefly:

- **Project goals** — two or three lines. What it is, who for, what's out of scope right now.
- **Architecture overview** — the shape in a few lines, then a link to `docs/ARCHITECTURE.md`. Enough to know where a file probably lives.
- **Design / style guide** — code conventions and, if there's a UI, the visual rules that keep it coherent.
- **Constraints and policies** — the things that would be expensive to violate. Cost ceilings, what must never be committed, data that can't leave the machine, dependencies not to add.
- **Repository etiquette** — branch naming, commit style, whether to PR or push, what needs review.
- **Frequently used commands** — dev server, tests, lint, build, deploy. The actual commands, copy-pasteable.
- **Documentation map** — one line each pointing at the docs above. This is what keeps CLAUDE.md short: it doesn't explain the architecture, it says where the architecture is explained.

Keep out: anything that duplicates a spoke document, anything that's already obvious from the code, and history. The changelog is for history.

**Keeping it current** is the part that usually fails. Two habits make it work: `/update-docs-and-commit` reviews CLAUDE.md as part of every commit, and hitting a milestone is an explicit trigger to reread it top to bottom and delete what's no longer true. Deleting is the maintenance work — additions take care of themselves.

## The document set

**`docs/project_spec.md`** — the signed-off output of Phase 1. It changes when the *product* changes, not when the code does. When it changes, note it in the changelog, because a spec that drifts silently is how a project ends up building something nobody agreed to.

**`docs/ARCHITECTURE.md`** — two halves. *System overview*: the components, the request flow end to end, the external dependencies, a diagram if it's non-trivial. *Component architecture*: per component, what it owns, its interface, what it depends on, and the decisions that would be annoying to reverse. Update when a component is added, removed, or changes responsibility — not for every file.

**`docs/CHANGELOG.md`** — reverse-chronological, grouped by version or date, in the Keep a Changelog shape (Added / Changed / Fixed / Removed). Written for a human skimming for "when did that break?", so entries describe the user-visible or developer-visible change, not the diff. `/update-docs-and-commit` appends to this on every commit.

**`docs/PROJECT_STATUS.md`** — the "where are we" document, and the first thing to read at the start of a session. Three parts: the milestone table from the spec with status per row, what's been accomplished (with dates), and what's next (a short ordered list, not a backlog). Keep "next" to three to five items — a status doc that's also a backlog stops being a status doc.

**`docs/reference/<feature>.md`** — one per key feature, created when a feature is complex enough that someone would otherwise reverse-engineer it from code. Covers what it does, how it works, the API or interface, config, gotchas. Create these as features land, not upfront.

**Suggest more as the project needs them.** Common additions worth proposing when the shape of the project calls for it: `docs/API.md` for a service with external consumers, `docs/DATA_MODEL.md` once the schema is non-obvious, `docs/DEPLOYMENT.md` at first deploy, `docs/DECISIONS.md` (lightweight ADRs) when the project starts accumulating "why did we do it that way", `docs/reference/prompts.md` for anything with substantial LLM prompt engineering. Propose them at the moment the need appears rather than creating empty files during setup.

## .env and .env.example

Derive both from the chosen stack. Every external service in the engineering design that needs credentials gets a variable.

- `.env.example` is committed, with every key present and dummy or empty values, plus a comment saying where to get each one. It's documentation as much as config.
- `.env` is real values and is **never** committed — confirm `.gitignore` covers it before the first commit, and check it again before pushing. A leaked key in git history is a genuinely bad afternoon.
- Group by service with comments, and mark which are required for the app to boot versus optional.
- If the project deploys somewhere, note in `.env.example` where production values live (Vercel project settings, VM env file, secrets manager) so it's not a mystery at deploy time.

Never write real secret values into any file the user hasn't explicitly handed you, and never echo a secret back into chat.
