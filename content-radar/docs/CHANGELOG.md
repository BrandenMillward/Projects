# Changelog

All notable changes to this project. Newest first.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — 2026-08-08

### Added

- **Content radar pipeline.** `radar fetch` polls RSS/Atom feeds, the Hacker News
  Algolia API and the arXiv API concurrently (8 workers, 10s timeout), appending
  to `state/items.jsonl`. No API keys required.
- **Clustering.** Canonical-URL dedupe plus fuzzy title matching
  (`token_set_ratio ≥ 85`) within a lane, so one announcement covered by eight
  outlets becomes one candidate.
- **Title stemming** in `normalise.py`, so headline tense and number stop
  splitting a story — "OpenAI ships X" and "X shipped by OpenAI" now cluster.
- **Scoring.** Five weighted components (recency, source trust, engagement,
  authority, corroboration) with authority weighted highest at 0.30, plus a 0.15
  demotion for anything already surfaced.
- **State.** Append-only JSONL for items, the seen-set and decisions, committed to
  git; 180-day pruning on the seen-set, 90-day suppression for killed topics.
- **Delivery.** `notify.py` with a `Channel` interface, a console channel, a
  Telegram channel, and a payload written for the session to relay via
  `PushNotification` / `SendUserFile`. The payload is parsed from the committed
  brief so the message and the repo cannot drift.
- **Failure alerts** on three triggers: run raises, >30% of sources fail, or
  fewer than 3 topics.
- **`/radar`** slash command carrying the editorial criteria, and **`/video`**
  plus the `video-drafts` skill for the two-cut YouTube + TikTok script package.
- 29 sources across `ai`, `regulated` and `genfinance` lanes.
- 46 offline tests.

### Notes

- **Reddit is deliberately excluded as a source.** Unauthenticated `.json`
  endpoints have returned 403 since 28 May 2026 and OAuth registration is
  approval-gated. The generational-finance lane uses MoneySavingExpert, Which?,
  the IFS, Pensions Age, This is Money and GOV.UK instead.
- The editorial layer runs in-session rather than through the Anthropic API, so
  the project needs no `ANTHROPIC_API_KEY` and adds no per-token cost.
- `radar fetch` has not yet run against live sources — the build environment
  blocks outbound egress to feed hosts. See `docs/PROJECT_STATUS.md`.
