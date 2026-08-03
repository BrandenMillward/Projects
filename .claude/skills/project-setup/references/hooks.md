# Hooks

Hooks are shell commands the harness runs at fixed points in the session lifecycle. The key property: **the harness runs them, not the model.** That's what makes them the right tool for anything that must happen every time regardless of what Claude decides — a formatter that always runs, a guard that always blocks. An instruction in CLAUDE.md is a suggestion; a hook is a rule.

## The events

| Event | When it fires | Example use |
|---|---|---|
| `PreToolUse` | Before any tool is executed | Block dangerous commands |
| `PostToolUse` | After a tool completes successfully | Run linters, format code |
| `PermissionRequest` | When the user sees a permission dialog | Auto allow / deny |
| `Notification` | When Claude sends notifications | Slack / Discord / SMS alerts |
| `UserPromptSubmit` | When the user submits a prompt | Validate prompts, inject context |
| `Stop` | When Claude finishes responding | Check tests |
| `SubagentStop` | When a subagent finishes | Same as Stop, for agents |
| `PreCompact` | Before context compaction | Save transcript snapshot |
| `SessionStart` | When a session begins or resumes | Set env vars |
| `SessionEnd` | When a session ends | Logging, cleanup |

## Recommended starter set

Recommend these three to nearly every project, because the payoff is immediate and the failure mode is mild:

**1. Format on write** — `PostToolUse` on `Edit|Write`, running the project's formatter on the changed file (`prettier --write`, `ruff format`, `gofmt`). Removes formatting from code review permanently and costs nothing. Match the formatter to the chosen stack.

**2. Guard destructive commands** — `PreToolUse` on `Bash`, exiting non-zero to block the genuinely unrecoverable: `rm -rf` on anything outside the project, force-push to the default branch, dropping a database, `git clean -fdx`. Keep the blocklist short and specific. A guard that fires on safe commands gets disabled within a week, which is worse than not having one.

**3. Session status** — `SessionStart`, printing current branch, git status summary, and the "what's next" section of `docs/PROJECT_STATUS.md`. Cheap, and it's what makes picking up a project after two weeks away not require archaeology.

**Recommend based on the project:**

- **Secret scan before commit** — `PreToolUse` on `Bash(git commit*)`, blocking if staged files contain anything matching an API-key pattern or if `.env` is staged. Strongly recommended for anything with paid API keys; a leaked key is the most likely expensive mistake in a solo project.
- **Test check on stop** — `Stop`, running the test suite and reporting failures. Good once tests exist and the suite is fast; actively harmful if it's slow, since it delays every single response. Ask about suite runtime before recommending it.
- **Typecheck on write** — `PostToolUse` on `Edit|Write` for `.ts`/`.tsx`, running `tsc --noEmit` on the project. Worth it for TypeScript projects past trivial size.
- **Notification to phone/Slack** — `Notification`, for long-running work the user walks away from.
- **Transcript snapshot** — `PreCompact`, saving the transcript before it's compacted. Useful when sessions run long and the retro agent needs the history.
- **Session log** — `SessionEnd`, appending a line to a local log. Low value alone; pairs well with the retro agent.

Skip anything that doesn't clearly earn its latency. Every `PreToolUse` and `PostToolUse` hook runs on *every* matching tool call, so a slow hook is a tax on the entire session — that's the tradeoff to name when recommending each one.

## Setting them up

Hooks live in `.claude/settings.json` (committed, shared with the project) or `.claude/settings.local.json` (personal, gitignored). Project-wide rules like formatting and command guards belong in the committed file; anything with a personal path or a webhook URL belongs in the local one.

`assets/hooks/settings.example.json` has working examples of each recommended hook. Copy the relevant blocks, substitute the project's real commands, and delete the rest — a settings file full of commented-out hooks is a settings file nobody edits confidently.

Two things to do after writing them:

- **Test each one before declaring it done.** Trigger the event and confirm the hook fires and does what's intended. A hook that silently fails is worse than no hook, because the user now believes they're protected.
- **Tell the user what you installed, in plain terms** — what fires, when, and how to turn it off. Hooks act invisibly, so an unexplained hook feels like the tool malfunctioning the first time it blocks something.

If the user wants hooks configured beyond this starter set, the `update-config` skill handles `settings.json` edits and knows the current schema.
