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

1. **Run `uv run radar fetch --since 7` somewhere with open egress.** Fix whichever
   feed URLs 404 or 403 from the failure report. This is the one thing standing
   between MVP-built and MVP-working.
2. **Run `/radar` end to end for real** and check the brief is worth reading —
   the scoring weights are a first guess and will need tuning against real items.
3. **Create the weekly Routine** (`create_new_session_on_fire: true`,
   `notifications: {push, email}`, Monday morning at an off-peak minute).
   Verify the first unattended firing actually delivered before trusting it.
4. **Add Telegram** once the brief is proving useful: @BotFather → token + chat
   ID → `.env`. Roughly five minutes.
5. **Draft one video package** with `/video` and see whether the two-cut format
   survives contact with an actual recording.
