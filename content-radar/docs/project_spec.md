# Content Radar — Project Spec

> Status: agreed | Last updated: 2026-08-08
> This is the source of truth for what we're building. Code, architecture, and docs derive from it.

---

## 1. Project Requirements

### Purpose

A weekly content radar that polls ~29 AI, tech and finance sources, clusters and scores what it finds against Branden's authority and publishing history, and delivers a brief proposing 3–5 topics worth writing — each with a pitch, an angle, a why-you and sources. On request it renders a chosen topic into a YouTube explainer script and a TikTok/Shorts vertical cut.

### Who it's for

Branden, solo. AI Orchestration Architect at Intent HQ; writes for senior data/AI practitioners and engineering/analytics leaders at `brandenmillward.github.io`.

Today he either writes when a topic happens to occur to him, or runs the `blog` skill cold and picks from three angles invented on the spot. Nothing records what was considered last week, and nothing tells him what is actually moving in his field right now.

### Problem it solves

Two failures of the status quo.

**Nothing accumulates.** The `blog` skill's step 1 proposes three angles and discards them the moment the session ends. There is no topic queue anywhere — the published archive is the only durable state — so the skill will happily re-pitch a theme that was rejected a fortnight ago.

**Nothing filters.** General AI news is abundant and mostly worthless to him. The scarce judgement is not "is this newsworthy" but "does he specifically have standing to write about it". A radar that returns 200 headlines has solved the wrong problem.

### What it does

- **Fetch.** `radar fetch` polls sources declared in `radar/feeds.toml` — RSS/Atom feeds, the Hacker News Algolia API, and the arXiv API — appending raw items to `state/items.jsonl`. Each source carries a lane (`ai` / `regulated` / `genfinance`) and a trust weight. No API keys are required.
- **Cluster.** Items dedupe by canonical URL (lowercased host, `utm_*`/`ref` stripped, fragment dropped, trailing slash normalised) then cluster by normalised-title similarity (`rapidfuzz.token_set_ratio ≥ 85`) within a lane. Titles are stemmed first so "OpenAI ships X" and "X shipped by OpenAI" agree. Eight articles about one announcement become one candidate whose cluster size is itself a corroboration signal.
- **Score.** Each cluster gets a deterministic score in [0, 1] from five weighted components — recency (48h half-life), source trust, engagement (log-scaled HN points), **authority match against Branden's actual expertise (weighted highest at 0.30)**, and corroboration — with a 0.15 multiplier applied to anything already in `state/seen.jsonl`. This is arithmetic, not judgement.
- **Judge.** `/radar` hands the top ~30 clusters to Claude in-session, which applies what the score cannot: does this intersect multi-agent orchestration, guardrails, XAI in regulated industries, or AI upskilling? Is there a counter-position, or would it restate the consensus? Has he covered it — checked against `state/decisions.jsonl` and the live archive at `brandenmillward.github.io/blog/index.html`?
- **Brief.** Writes `briefs/YYYY-MM-DD.md`: 3–5 topics, each with a one-line pitch, the angle, why Branden specifically, source URLs and a suggested lane — plus a "deliberately not suggesting" section naming what was dropped and why. That section is what stops week five repeating week two.
- **Deliver.** The brief is **pushed, not parked.** Committing markdown is the archive, not the delivery. Every run sends the full topic list — every topic with pitch and why-you, so the week can be judged without opening anything — via phone push, email, and (v1) Telegram, and surfaces the brief file itself. A run that crashes, loses more than 30% of its sources, or yields fewer than 3 topics notifies too: a silently broken radar and a quiet news week must never look identical.
- **Decide.** `/radar pick|kill|shipped <n>` appends to `state/decisions.jsonl`. Killed topics are suppressed 90 days; shipped ones permanently.
- **Render — blog.** `/radar blog <n>` hands the topic, angle and sources to the existing `blog` skill as its topic argument. The drafter is not reimplemented.
- **Render — video.** `/video <n>` writes `drafts/<slug>-video.md` with **two cuts of one research spine**: a YouTube explainer (5–8 min, ~900–1,200 words VO — 3 titles, thumbnail brief, 0–15s hook, 4–6 beats each with VO + on-screen text + b-roll prompt, close, description, chapters, tags) and a TikTok/Shorts vertical (45–60s, ~130–160 words — a hook landing inside 3 seconds, 3 beats with upper-third on-screen text, caption, hashtags).

### Jobs to be done

1. When I sit down to write, I want 3–5 topics I have standing to write about already researched and sourced, so I can start drafting instead of deciding what to draft.
2. When a topic keeps resurfacing across weeks, I want to see that it's recurring rather than re-evaluating it cold each time, so I can tell a real trend from noise.
3. When I want short-form video, I want a chosen topic cut into a YouTube explainer and a vertical script in my voice, so I'm editing a draft rather than facing a blank page.

---

## 2. Milestones

| Version | Core app functionality |
|---|---|
| **MVP** | • Single user, local, **no keys required**<br>• **Core function:** `radar fetch` + `/radar` produce `briefs/YYYY-MM-DD.md` with 3–5 sourced, angled topics<br>• **Delivered, not parked:** full topic list via phone push + email, brief file surfaced in-session<br>• RSS + HN Algolia + arXiv sources<br>• `state/seen.jsonl` dedupe so week two doesn't repeat week one |
| **v1** | • **Scheduled:** weekly Routine fires a fresh session, generates, delivers and commits unattended<br>• **Telegram channel** via the notifier interface (first secret; ~5 min @BotFather setup)<br>• **Failure alerts:** crash, >30% source loss, or <3 topics all notify<br>• **Decisions:** `/radar pick` records picked/shipped/killed; killed suppressed 90 days<br>• **Video packages:** `/video <n>` writes the two-cut draft<br>• **Blog hand-off:** `/radar blog <n>` into the existing `blog` skill |
| **v2** | • **Reply to pick:** answer the Telegram message with a number to record the decision<br>• **Archive awareness:** fetch the live blog index and penalise covered topics automatically<br>• **Lane tuning:** per-lane weights adjusted from decision history |
| **Later** | • Embedding-based clustering replacing fuzzy title matching<br>• Higgsfield render hand-off from the script package<br>• Performance loop: which suggested topics actually got published |
| **Not in scope for now** | User accounts, auth, payments, multi-user, a web UI, any hosted service, automatic publishing to the live site, auto-posting to YouTube/TikTok, and **Reddit as a source** (see below) |

### Why Reddit is excluded

Reddit deprecated unauthenticated `.json` endpoints on **28 May 2026** — they now return 403, which silently broke most open-source Reddit tooling. Self-service OAuth registration is closed; new tokens require approval under the Responsible Builder Policy, and unauthenticated access is capped at 10 req/min where it works at all. Reddit was the obvious source for the generational-finance lane; it is replaced by MoneySavingExpert, Which?, the IFS, Pensions Age, This is Money and GOV.UK.

---

## 3. Engineering Design

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Already present; matches the existing repo's idiom |
| Packaging | `uv` + `pyproject.toml` | `uv` already installed; no existing dependency file to conflict with |
| Ingestion | `feedparser`, `httpx` | Tolerant feed parsing; keyless HN Algolia + arXiv |
| Dedupe | `rapidfuzz` | Fast trigram similarity, no model weights |
| Storage | JSONL in git | Diffable, greppable, reviewable in a PR, readable on a phone |
| Editorial | Claude Code in-session | No key, no SDK, no extra billing; taste lives where taste belongs |
| Scheduling | Claude Routine, fresh session per fire | Durable and server-side; **notifications only work on fresh-session Routines** |
| Delivery | Push + email (MVP), Telegram (v1) | The brief has to arrive; a committed file is an archive |

**The editorial layer does not call the Anthropic API.** The brief is generated by `/radar` running inside a Claude Code session, so the judgement layer *is* the session: no `ANTHROPIC_API_KEY`, no SDK dependency, no per-token billing on top of the subscription, and no second place where prompts live. Python stays deterministic and testable; Claude does the part that needs taste.

**`CronCreate` was considered and rejected** for scheduling — it is session-only, in-memory, and self-deletes after 7 days, so it cannot hold a weekly cadence.

### Delivery channels

| Channel | Carries | Keys | Milestone |
|---|---|---|---|
| Routine push + email on completion | Full topic list; email persists as a to-do | none | MVP |
| `PushNotification` from inside the run | One line, <200 chars — lead topic + count | none | MVP |
| Brief file surfaced in-session | The markdown itself | none | MVP |
| Telegram `sendMessage` | Full topic list; the only **two-way** channel | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | v1 |

Telegram is free at this volume (one weekly message against a 30 msg/sec free tier). Its cost is structural, not financial: it introduces the project's first secret. MVP therefore stays keyless, but `notify.py` defines a `Channel` interface from day one so Telegram lands as a config entry rather than a refactor. Being two-way is what later makes "reply `2` to pick topic 2" possible.

### Engineering requirements

- A full `radar fetch` across ~29 sources completes in **under 60 seconds**, 10s per-source timeout, concurrency capped at 8.
- **One dead source never fails a run.** Fetch errors are caught per-source and surfaced in the brief.
- **Failure is loud, on three triggers:** the run raises; **more than 30% of sources fail**; or the brief would carry **fewer than 3 topics**. Each notifies with the specific cause.
- **Delivery is not best-effort.** A run is complete only when at least one channel confirms. If every channel fails the run exits non-zero — the brief still commits, but the run is not reported as successful.
- Notification content is generated **from the committed brief**, so the phone and the repo cannot drift apart.
- Every source is polled with a descriptive `User-Agent`. Feeds and public read-only APIs only — no HTML scraping of article bodies, no paywall circumvention.
- **No secrets in MVP.** `.env` is gitignored; a token never enters a brief, a log line or a commit.
- `state/seen.jsonl` stays small via 180-day pruning.
- Every claim in a brief carries a source URL. Claude adds angles; it never invents a fact.

### Architecture

See `docs/ARCHITECTURE.md` for the component breakdown and a traced request.

### System design

- **Data model.** `items.jsonl`: `{url, canonical_url, title, source, lane, published_at, points, summary, fetched_at}`. `candidates.json`: `{cluster_id, title, urls[], sources[], lane, size, score, components{}, first_seen}`. `decisions.jsonl`: `{at, cluster_id, title, state, reason}`.
- **External contracts.** HN Algolia `GET /api/v1/search_by_date`; arXiv `GET /api/query`; RSS/Atom via `feedparser`. All keyless, all read-only.
- **Where state lives.** Entirely in `content-radar/state/`, committed. No database, no server.
- **Notifier contract.** `Channel.send(payload) -> bool`. A channel that raises is caught and skipped; the run fails only if every channel fails.

---

## Open questions

- **Live fetch is unverified.** The sandbox this was built in blocks outbound egress to all feed hosts (403 at the proxy), so `radar fetch` has never run against real sources. Clustering, scoring, state and delivery are verified against fixtures. The feed URLs in `feeds.toml` are best-effort and some will need correcting on first real run — the per-source failure report exists precisely to make that a five-minute fix.
- **Higgsfield availability.** Its `faceless-channel-video` workflow is the natural target for the Later rendering milestone, but the MCP server disconnected during planning, so credits and presets are unverified. Not an MVP dependency.
- **Blog-repo hand-off.** Currently write-files-and-copy. If that becomes tedious, attaching `brandenmillward.github.io` to a session and opening a PR is the upgrade.
