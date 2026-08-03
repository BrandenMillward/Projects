# Plugins and MCP Servers

Two separate walkthroughs. The user asked to be *walked through* both, which means the deliverable is understanding, not a config file — for each one they should end up knowing what it does, why this project would want it, and what it costs them (auth, money, latency, context).

## Don't recite a catalog from memory

Available plugins and MCP servers change constantly. Query live:

- `SearchPlugins` / `ListPlugins` — what's installable and what's already installed
- `SearchMcpRegistry` — MCP servers in the registry
- `ListConnectors` / `ListMcpResourcesTool` — what's already connected in this environment

Check what's already installed before recommending anything. Recommending a server the user already has is a fast way to look like you're not paying attention. If the tools aren't available in the session, say you can't enumerate them live, recommend from the categories below, and tell the user to check `/plugin` and `/mcp`.

## How to run each walkthrough

Go one at a time, and for each candidate give three things: **what it does**, **why this project specifically wants it** (tie it to a requirement or milestone — if you can't, that's a sign to drop it), and **what it costs** (an account, an API key, a paid plan, or context budget). Once the user has heard all the pitches, ask once — a single `AskUserQuestion` with `multiSelect: true` and one option per candidate, so they pick the set. That reads as a menu, which is what it is; four separate yes/no questions reads as an interrogation and gets rubber-stamped.

Then be honest about the ceiling: **every MCP server's tool definitions consume context in every session.** Five servers is usually fine; fifteen measurably degrades performance and crowds out the actual work. Recommend a small set and tell the user they can add more later — this is the advice most setup guides skip, and it's the one that matters most six weeks in.

## Plugins — what to consider by project shape

Match to the project rather than installing broadly.

- **Any project** — the commit/PR workflow plugins, if the user isn't getting those from the custom slash commands created in step 7. Don't install a plugin that duplicates a command you just wrote; pick one or the other and say which.
- **Web / frontend** — browser automation and visual testing, accessibility checking, component library helpers.
- **Backend / API** — API testing and client generation, database and migration tooling.
- **Data / ML** — notebook tooling, experiment tracking, dataset helpers.
- **Anything with a deploy target** — the hosting provider's plugin, if one exists.
- **Code quality** — review, security scanning, and test-generation plugins. Worth it once there's enough code to review; premature on day one.

A plugin the user won't use is worse than no plugin, because it still shows up in every session. When in doubt, skip it and revisit at the next milestone.

## MCP servers — what to consider

- **GitHub** — issues, PRs, CI status, code search without leaving the session. Nearly always worth it once the repo exists.
- **The database** (Supabase, Mongo, Postgres) — schema inspection, migrations, and querying during development. High value for anything data-backed; make sure it's pointed at a dev project, not production.
- **The deploy platform** (Vercel, cloud provider) — deploy status, build logs, runtime errors. Worth it from the first deploy, not before.
- **Browser / Playwright** — real end-to-end checks against a running app. Pairs with a frontend-testing subagent; recommend them together or neither.
- **Docs / search servers** — pulling current framework documentation instead of relying on training data. Valuable when the stack has moved recently.
- **Design tools** (Figma, tldraw) — only if there's an actual design source to pull from.

For each accepted server, tell the user concretely what it needs — OAuth flow, an API token and where to generate it, a config entry — and whether it's per-project or global. Add project-scoped servers to `.mcp.json` in the repo so the config travels with the project; leave personal/global servers to the user's own config. Never write a real token into a committed file: reference it as `${VAR_NAME}` and add the variable to `.env.example`.

Record the final set — chosen, skipped, and why — in `docs/PROJECT_STATUS.md` or a short `docs/reference/tooling.md`. The "why skipped" list is the useful half: it stops the same conversation happening again in three weeks.
