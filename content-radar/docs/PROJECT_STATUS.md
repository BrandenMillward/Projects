# Project Status

> Last updated: 2026-08-08

## Milestones

| Version | Status | Notes |
|---|---|---|
| **MVP** | **Mostly done** | Pipeline, scoring, state and delivery built and verified against fixtures. `radar fetch` is unverified against live sources — see blocker below. |
| **v1** | Not started | Routine, Telegram, decisions CLI wiring, video command exercised for real |
| **v2** | Not started | Reply-to-pick, automatic archive check, lane tuning |
| **Later** | Not started | Embedding clustering, Higgsfield render, performance loop |

## Accomplished (2026-08-08)

- Planning interview and signed-off spec (`docs/project_spec.md`).
- Verified two facts that changed the design: **Reddit's unauthenticated `.json`
  endpoints have returned 403 since May 2026** (so Reddit is out and the
  generational-finance lane is served by MSE, Which?, IFS, Pensions Age, This is
  Money and GOV.UK); and **Routine push/email notifications only work on
  fresh-session-per-fire Routines**, which fixed the scheduling shape.
- Built `radar`: `normalise`, `cluster`, `score`, `state`, `sources`, `notify`, `cli`.
- 29 sources declared across three lanes in `radar/feeds.toml`.
- **46 tests passing**, all offline.
- Verified with fixtures: an 11-item / 4-outlet fixture collapsed to 6 clusters,
  ranked signal 6× above noise, and — the week-two test — previously-surfaced
  topics dropped from 0.84 to 0.13 once marked seen.
- Verified the thin-week failure path exits non-zero with a specific reason.
- `/radar` and `/video` commands, and the `video-drafts` skill.

## Blocked

- **Live fetch unverified.** The build sandbox blocks outbound egress to every
  feed host (403 at the egress proxy, an organisation policy denial — not
  something to route around). `radar fetch` has therefore never run against real
  sources. Some feed URLs in `feeds.toml` will be wrong; the per-source failure
  report exists to make that a quick fix on the first real run.

## What's next

1. **Trigger the workflow manually** — Actions tab → *Weekly fetch* → **Run
   workflow**. GitHub runners have open egress, so this is the first real test of
   `radar fetch` and the fastest way to find out which feed URLs are wrong. The
   run will go red if more than 30% fail; the `fetch-log` artifact names each one.
2. **Fix the failing feeds** in `radar/feeds.toml` from that log, and re-run until
   the job is green. This is the one thing standing between MVP-built and
   MVP-working.
3. **Run `/radar` end to end** against the committed `state/candidates.json` and
   judge whether the brief is worth reading. The scoring weights are a first
   guess — `authority` is deliberately the heaviest at 0.30, and that is the knob
   to turn first if the output feels wrong.
4. **Create the weekly Routine** for the editorial half
   (`create_new_session_on_fire: true`, `notifications: {push, email}`). Only
   after step 2 is green — a Routine whose first unattended firing is also its
   first real test will just deliver a failure notice.
5. **Add Telegram** once the brief is proving useful: @BotFather → token + chat
   ID → `.env` and repo secrets. Roughly five minutes.
6. **Draft one video package** with `/video` and see whether the two-cut format
   survives contact with an actual recording.

## Where things run

| Half | Runs on | Needs |
|---|---|---|
| `fetch` → `score` → `prune` | GitHub Actions, Monday 06:07 UTC | Open egress (runners have it) |
| `/radar` editorial, `/video` | A Claude session | This repo; no egress for the scoring input |

The split exists because the Claude environment this was built in blocks outbound
egress to feed hosts. Actions sidesteps that entirely: it commits the shortlist,
and the editorial half reads the repo rather than the internet.
