# project-setup

A Claude Code skill that takes a rough project idea and turns it into a planned, specified,
scaffolded repo.

**Phase 1 — Planning:** brainstorm, a 3–5 round interview, and a `docs/project_spec.md` with
specific requirements, jobs-to-be-done, capability-based milestones, and an engineering design
backed by a live tech-stack research report. Gated on your sign-off.

**Phase 2 — Setup:** GitHub repo, `.env` / `.env.example`, `CLAUDE.md`, the documentation set
(spec, architecture, changelog, status, per-feature references), `docs/STANDARDS.md`, plugin
and MCP server walkthroughs, slash commands, subagents, and hooks.

**Standards:** setup decides the coding standard and its toolchain, walks the ISO/IEC 25010
quality characteristics to turn vague engineering requirements into numbers, and asks once
whether any formal standards regime applies. For most projects the answer is no, and
`docs/STANDARDS.md` says so in a line. The sibling skill `project-change` reads that file to
decide which gates each later change must clear.

## Install

This skill needs to be available from *outside* any project, since its whole job is starting
new ones. Install it globally:

```bash
mkdir -p ~/.claude/skills
cp -r .claude/skills/project-setup ~/.claude/skills/
```

It also works project-locally from where it sits in this repo, which is useful for editing it.
If you have both, the project-scoped copy wins inside this repo.

## Use

Say what you want to build. It triggers on new-project language without needing to be named:

- "I want to build a tool that turns my voice notes into blog drafts"
- "new project: a CLI for tracking climbing sessions"
- "/project-setup"

To run only part of it, say so — "just do the planning phase" or "skip hooks for now" — and it
will stop where you tell it to and note what was skipped in `docs/PROJECT_STATUS.md`.

## Layout

```
SKILL.md                       orchestrator: phases, gates, step order
references/
  planning-interview.md        question banks, the specificity standard
  engineering-design.md        research report format, stack menu
  coding-standards.md          house baseline, toolchain per stack, enforcement ladder
  iso-standards.md             does any standard apply, and the clause-to-artifact map
  documentation-system.md      CLAUDE.md + doc set, what goes where
  plugins-and-mcp.md           how to run both walkthroughs
  commands-and-agents.md       slash command vs subagent, what to create
  hooks.md                     the 10 events, recommended starter set
assets/
  templates/                   project_spec, CLAUDE.md, ARCHITECTURE, CHANGELOG,
                               PROJECT_STATUS, STANDARDS, reference-doc, env.example
  commands/                    commit, commit-push-pr, update-docs-and-commit,
                               create-issues, standards-check
  agents/                      changelog-writer, frontend-tester, retro
  hooks/settings.example.json  tested hook configs
scripts/scaffold.py            lays down the file skeleton
```

## scaffold.py

The skill runs this for you, but it's usable standalone:

```bash
python3 scripts/scaffold.py --name "Thumbnail Studio" --dir ./thumbnail-studio \
  --stack next-supabase --one-liner "Generate YouTube thumbnails from a short prompt." \
  --agents all --hooks
```

Templates land with `{{PLACEHOLDER}}` tokens and guidance comments still in them — the agent
fills those in from the signed-off spec. Existing files are never overwritten without
`--force`, and `--dry-run` shows what would happen. `--stack` accepts `next`, `next-supabase`,
`vue`, `angular`, `express`, `node`, `fastapi`, `django`, `python`, `swift`, `fullstack`;
unknown values just get every `.gitignore` set, which is harmless.

## Customising it

The two files worth editing for your own taste:

- `references/engineering-design.md` — the "house menu" table is your default stack shortlist.
- `references/planning-interview.md` — the question banks and the specificity examples.
- `references/coding-standards.md` — the house code baseline and the per-stack toolchain table.

## Related

`project-change` governs what happens after the baseline exists — impact assessment,
reversibility, standards gates, and change records that diff against this spec rather than
replacing it.

`assets/hooks/settings.example.json` ships with a `_README` key and `_comment` keys explaining
each hook; strip those when copying into a real `.claude/settings.json`.
