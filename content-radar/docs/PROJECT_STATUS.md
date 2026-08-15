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

Nothing. The first live run cleared the one outstanding blocker.

## First live run — 2026-08-14

Ran green on GitHub Actions in 29 seconds: **2,677 items from 23 of 29 sources.**
That verified the whole chain end to end for the first time — fetch, cluster,
score, prune, and the commit-back — on real data rather than fixtures.

Six sources failed, and the fixes were not all "correct the URL":

| Source | Failure | Outcome |
|---|---|---|
| Anthropic Engineering | 404 | **Dropped** — no official RSS feed exists, only community scrapers |
| ICO News | 404 | **Dropped** — ICO retired its feeds in a site redesign |
| FT Adviser | 404 | **Dropped** — no public feed URL could be verified |
| Pensions Age | 404 | **Dropped** — same |
| Institute for Fiscal Studies | 403 | **Replaced** — refuses automated clients |
| MoneySavingExpert | 403 | **Replaced** — same |

The two 403s could be reached by sending a browser User-Agent, but that means
circumventing a deliberate block, so they were replaced rather than spoofed.

**The failures were concentrated in the generational-finance lane** — three of
its six sources — while the AI lane lost one of fifteen. Replacements: Guardian
Money, BBC Business and a GOV.UK pensions keyword feed, plus a GOV.UK data
protection feed to backfill the ICO. Now 27 sources: 14 AI, 7 regulated,
6 generational finance.

## What's next

1. **Re-run the workflow** after the feeds.toml fix and confirm 27/27 sources.
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
