---
name: project-setup
description: Run the full new-project bootstrap — a planning interview that produces a specific, testable project_spec.md (requirements, jobs-to-be-done, capability milestones, engineering design with a tech-stack research report), then the scaffolding — GitHub repo, .env, CLAUDE.md, the automated documentation set, plugins, MCP servers, slash commands, subagents, and hooks. Use this whenever the user is starting a new project, spinning up a new repo, says "new project", "set up a project", "scaffold this", "help me plan an app/tool/service I want to build", brings a project idea they want turned into a spec, or asks for project docs/CLAUDE.md/slash commands/subagents/hooks to be set up from scratch. Use it even when they only describe the idea and don't say the word "setup" — a fresh idea with no repo yet is exactly what this is for.
---

# Project Setup

Take a rough project idea and turn it into a planned, specified, scaffolded repo that Claude Code can work in productively from day one.

Two phases, and the gate between them matters: **plan fully, get sign-off, then build.** Scaffolding a repo before the spec is settled means every file — CLAUDE.md, the architecture doc, the slash commands, the hooks — encodes guesses that get stale immediately. The spec is the source of truth everything else is generated from.

## Phase 0 — Orient (2 minutes, no files yet)

Before asking anything, establish where you are:

1. **Read back what you heard.** Restate the user's idea in 2–3 sentences and say what you think the core loop of the product is. Getting corrected here is cheap; getting corrected after writing a spec is not.
2. **Check the ground.** Is there an existing directory? A git repo? Run `git status` / `ls`. A new project inside an existing repo is a different job from a fresh repo — ask which it is if it's ambiguous.
3. **Set expectations.** Tell them the shape: brainstorm → interview → spec → sign-off → scaffold. Say roughly how many question rounds to expect (usually 3–5). People answer better questions when they know the process has an end.

Then brainstorm *before* interrogating. Offer 3–5 angles they might not have considered — adjacent use cases, a sharper wedge, a version of the idea that's half the scope, a risk that could kill it. This is the highest-leverage part of the whole skill: a good brainstorm changes what gets built, while a good interview only records it. Give them opinions, not a menu.

## Phase 1 — Planning

Work through the interview and draft the spec section by section, showing each section as you finish it rather than dumping the whole document at the end. Read `references/planning-interview.md` now — it has the question banks, the specificity standard, and how to run the rounds.

The spec has three parts, built in this order:

**1. Project requirements** — purpose, functionality, jobs-to-be-done; who it's for, what problem it solves, what it does. The whole game here is specificity. `Users can create journal entries` is worthless — it survives any implementation and tests nothing. `Users create journal entries by first selecting a prompt and then responding to it; prompts are generated from their past entries; they can respond by writing or by recording video` is a spec — it names the entry point, the mechanism, and the modalities, and you can tell whether a build satisfies it. Hold every requirement to that bar and rewrite the vague ones in front of the user.

**2. Milestones** — capability-based versions, not dates. `MVP / v1 / v2 / Later / Not in scope for now`, where each row says what the product can *do* at that version. The "not in scope" row is load-bearing: it's the only thing that stops scope creep later, and it's where things like accounts, payments, and multi-user usually go for a solo build. See `assets/templates/project_spec.md` for the table shape.

**3. Engineering design** — the technical requirements for how it gets built: tech stack, engineering requirements, architecture, system design. Do not pick the stack for them silently. Read `references/engineering-design.md` — it covers the research report you produce first (with live lookups, not memory), the house stack menu, and how to present the choice.

**Gate:** write `docs/project_spec.md`, show it, and ask for explicit sign-off before touching Phase 2. If they want changes, revise and re-ask.

## Phase 2 — Setup

Everything here derives from the signed-off spec. Run `scripts/scaffold.py` to lay down the file skeleton, then fill each file with real content from the spec — the script writes structure and placeholders, you write substance. Never ship a file with `{{...}}` or `TODO` left in it unless the TODO is genuinely a note to the user.

```bash
python3 <skill-dir>/scripts/scaffold.py --name "<project-name>" --dir <project-dir> --stack <stack-slug>
```

Work through these in order. Each has a reference file with the detail; read it when you get to that step rather than all upfront.

| # | Step | Reference |
|---|------|-----------|
| 1 | GitHub repo — create it, set default branch, push the initial commit | below |
| 2 | `.env` + `.env.example` — derived from the chosen stack's services | `references/documentation-system.md` |
| 3 | `CLAUDE.md` — project memory, kept short and pointing outward | `references/documentation-system.md` |
| 4 | Documentation set — spec, architecture, changelog, status, feature refs | `references/documentation-system.md` |
| 5 | Plugins — walk through what's worth installing and why | `references/plugins-and-mcp.md` |
| 6 | MCP servers — same, with the auth/cost reality of each | `references/plugins-and-mcp.md` |
| 7 | Slash commands + subagents — tailored to this project | `references/commands-and-agents.md` |
| 8 | Hooks — recommend a starter set, explain the tradeoff of each | `references/hooks.md` |
| 9 | Commit and push | below |

### Step 1 — GitHub repo

Ask for the repo name (default: kebab-case of the project name) and public vs private before creating anything. Creating a repo is outward-facing and hard to undo quietly, so confirm rather than assume.

Use `mcp__github__create_repository` if the GitHub MCP tools are available, otherwise `gh repo create`. Then `git init` locally if needed, and set the remote. Don't push yet — push once at step 9 with a complete initial commit.

If the project lives inside an existing repo, skip creation and just confirm which branch to work on.

### Step 9 — Commit and push

One initial commit containing the whole scaffold, on a branch, with a message that says what the project is. Then `git push -u origin <branch>`. Only open a PR if the user asks.

## How to run this without being exhausting

The failure mode of a skill like this is a 40-question interrogation followed by a wall of generated files. Avoid it:

- **Ask one or two questions at a time, not four.** `AskUserQuestion` takes up to four, and filling the slots is a trap: people answer the first properly and the rest thinly, so you trade three real answers for one good one and three shrugs. Ask only what genuinely blocks the next step, and give a recommended option first — most people want a default they can override, not a blank slate.
- **Answer your own questions where you can.** If the project is obviously a web app, don't ask whether it needs a frontend. Infer, state the inference, and let them correct it. Every question you don't have to ask buys you one you do.
- **Show, don't ask, for structure.** Draft the milestone table and ask "what's wrong with this?" rather than asking them to invent it. Reacting is easier than generating.
- **Let them cut scope.** If they want to skip a step ("no hooks for now"), skip it and note it in `PROJECT_STATUS.md` under what's next. Don't relitigate.
- **Keep a running summary.** Between phases, restate decisions made so far in a few lines. Long interviews drift, and it's how you catch a contradiction before it's baked into the architecture.

## When the project already exists

If they run this on a repo that already has code, don't scaffold over it. Read the code first, then use the same phases to *recover* the spec — draft `project_spec.md` from what's actually there, show it, and let them correct it. Merge into existing `CLAUDE.md` and docs rather than overwriting. Ask before replacing any file that already has content.

## Files in this skill

- `references/planning-interview.md` — question banks, the specificity standard, running the rounds
- `references/engineering-design.md` — stack research report, house menu, architecture section
- `references/documentation-system.md` — CLAUDE.md and the doc set, what goes where, how they stay current
- `references/plugins-and-mcp.md` — how to walk through plugins and MCP servers
- `references/commands-and-agents.md` — choosing and writing slash commands and subagents
- `references/hooks.md` — hook events and a recommended starter set
- `assets/templates/` — file templates the scaffold script copies
- `assets/commands/`, `assets/agents/`, `assets/hooks/` — starter commands, subagents, hook config
- `scripts/scaffold.py` — lays down the directory and file skeleton
