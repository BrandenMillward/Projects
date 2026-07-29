# Slash Commands and Subagents

Both extend Claude Code, but they solve different problems, and picking the wrong one is the usual mistake.

| | Slash commands | Subagents |
|---|---|---|
| **Definition** | Shortcut to a prompt | Specialized agent for a specific task |
| **Best used for** | Quick tasks, automations | Parallel work, specialized tasks |
| **Context window** | Same context window | Fork of the context window |
| **Execution** | Synchronous in current session | Async, can run in parallel |
| **Examples** | `/architect`, `/commit`, `/run-tests` | Research, Plan, Search |

The practical test: **does this task need to know what we've been doing, or does it need to not pollute what we're doing?** A commit command needs the current session's context — it's summarizing work that just happened, so it's a slash command. A research task that reads forty files needs to keep those forty files *out* of the main context and hand back a conclusion, so it's a subagent.

## Choosing what to create

Start from the starter set in `assets/`, then add what this specific project needs. Ask the user which they want rather than installing all of them — an unused command is clutter, and the ones they'll actually use are usually obvious from how they described their workflow.

**Starter slash commands** (in `assets/commands/`, copied to `.claude/commands/`):

- `commit-commands:commit` — stage, write a conventional-commit message from the actual diff, commit
- `commit-commands:commit-push-pr` — the above, plus push and open a PR with a body derived from the changes
- `update-docs-and-commit` — the important one. Reviews the changed code, updates CHANGELOG / PROJECT_STATUS / ARCHITECTURE / CLAUDE.md as needed, then commits everything together. This is what keeps the documentation system from rotting, because docs updated in the same commit as the code are the only docs that stay true.
- `create-issues` — turn a milestone from the spec into GitHub issues, one per capability

**Worth proposing based on the project:**

- `/architect <feature>` — design a feature against the existing architecture before writing code. Good for anything with real structure.
- `/run-tests` — the project's actual test command plus triage of what failed
- `/deploy` or `/ship` — once there's a deploy target, encoding the real sequence and its preflight checks
- `/next` — read PROJECT_STATUS.md and start the next thing. Cheap to write, surprisingly good at removing session-start friction.
- `/review-diff` — self-review the working diff against the project's conventions before committing

**Starter subagents** (in `assets/agents/`, copied to `.claude/agents/`):

- `changelog-writer` — reads the diff and writes changelog entries in the project's voice. Forks context so the whole diff doesn't land in the main window.
- `frontend-tester` — drives the running app in a browser and reports what actually broke. Only worth creating if there's a UI *and* a browser MCP server or Playwright available; otherwise it's a stub that will disappoint.
- `retro` — at the end of a session, reviews what happened and proposes concrete improvements to CLAUDE.md, the commands, or the workflow. This is the compounding one: it's how the setup gets better instead of ossifying.

**Worth proposing based on the project:**

- `researcher` — reads docs and codebases and returns a conclusion, not a transcript
- `test-writer` — writes tests against a spec or a diff
- `security-reviewer` — for anything handling user data, auth, or payments
- `data-explorer` — for data projects: profile a dataset and report, without dumping rows into main context

## Writing them well

Both are markdown files with YAML frontmatter. Keep each one focused on a single job — a command that does four things is one nobody trusts, because they can't predict what it'll touch.

**Slash command** — `.claude/commands/<name>.md`:

```markdown
---
description: One line shown in the command list
allowed-tools: Bash(git*), Read, Edit
---

Imperative instructions. Reference arguments with $ARGUMENTS or $1, $2.
```

Scope `allowed-tools` to what the command actually needs. It's the difference between a command that runs without interrupting and one that prompts for permission every time.

**Subagent** — `.claude/agents/<name>.md`:

```markdown
---
name: agent-name
description: When to use this agent — this is what the model matches against, so be specific about the triggering situation
tools: Read, Grep, Glob
---

System prompt for the agent. Say what it does, what it returns, and what it must not do.
```

Two things make a subagent useful rather than annoying:

- **Give it the narrowest tool set that does the job.** A read-only research agent with write access will eventually write something.
- **Be explicit about the return format.** The subagent's context is discarded — only its final report survives, and the user never sees the report unless the main agent relays it. If it doesn't return the conclusion in a usable shape, all that work is wasted. Say what the report must contain.

After creating them, list what was created and what each is for, and tell the user they can invoke commands with `/<name>` and that subagents trigger automatically on matching work.
