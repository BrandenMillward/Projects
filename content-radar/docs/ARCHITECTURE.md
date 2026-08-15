# Architecture

## System overview

Five components. Four are deterministic Python; one is Claude in-session, and
that boundary is the main design decision in the project.

```mermaid
flowchart LR
  A[feeds.toml<br/>~29 sources] --> B[sources.py<br/>concurrent fetch]
  B --> C[state/items.jsonl]
  C --> D[cluster.py<br/>URL + fuzzy title]
  D --> E[score.py<br/>5 weighted components]
  E --> F[state/candidates.json<br/>top ~30]
  F --> G[/radar<br/>Claude editorial/]
  H[state/seen.jsonl<br/>state/decisions.jsonl] --> E
  H --> G
  I[live blog archive] --> G
  G --> J[briefs/YYYY-MM-DD.md<br/>committed = archive]
  J --> N[notify.py]
  N --> N1[phone push]
  N --> N2[email]
  N --> N3[brief file in-session]
  N --> N4[Telegram - v1]
  J --> K[/radar blog n/ --> blog skill]
  J --> L[/video n/ --> drafts/*-video.md]
```

### One run, traced end to end

The Routine fires `/radar` on Monday morning. The command runs
`uv run radar fetch --since 7`, which polls 29 sources across 8 threads and
appends ~400 items to `state/items.jsonl`; each source that fails is recorded
rather than raised, and if more than 30% fail the command exits non-zero and the
run stops with a failure notification instead of writing a misleading brief.

`radar score` canonicalises every URL, stems and normalises every title, collapses
near-duplicates into clusters within a lane, applies the five-component heuristic,
demotes anything already in `seen.jsonl` by a factor of 0.15, and writes the top
30 to `state/candidates.json` along with the suppressed-titles map.

Claude reads that file plus `decisions.jsonl` and the live blog index, discards
the candidates that are generic or already covered, and writes
`briefs/2026-08-10.md` with 3–5 topics — pitch, angle, why-you, sources, lane —
plus a "deliberately not suggesting" section.

`radar seen` marks those URLs surfaced. `radar notify` parses the committed brief,
builds the payload, sends on every available Python channel, and writes
`state/last_notification.json`; the session then relays it via `PushNotification`
and `SendUserFile`. Branden finds out because his phone told him, not because he
remembered to check the repo.

## Component architecture

### `normalise.py`
**Owns** the comparability of URLs and titles. `canonical_url` strips tracking
params, `www.`, fragments and default ports; `_stem` folds inflections so tense
and number don't split a story; `url_hash` is the seen-set key.
**Costly to reverse:** `url_hash` values are persisted in `seen.jsonl`. Changing
canonicalisation invalidates the entire seen-set, which would cause one week of
repeated suggestions.

### `sources.py`
**Owns** all network I/O and the source registry (`feeds.toml`).
**Interface:** `load_sources()`, `fetch_all(specs, since_days) -> FetchReport`.
**Decision:** every fetcher catches its own exceptions. A `FetchReport` carries
`ok`, `failed` and a `degraded` flag rather than raising, because a run that
loses one feed should still produce a brief and a run that loses a third should
not silently produce a thin one.

### `cluster.py`
**Owns** collapsing items into stories. Two passes: exact canonical URL, then
`token_set_ratio ≥ 85` within the same lane.
**Decision:** clustering never crosses lanes — the same headline in `ai` and
`genfinance` is two different stories for two different audiences.

### `score.py`
**Owns** the shortlist, not the selection. Five components summing to weight 1.0:
recency 0.22, source 0.18, engagement 0.15, **authority 0.30**, corroboration 0.15.
**Decision:** authority is weighted highest deliberately. "Is this newsworthy" is
abundant; "can Branden uniquely speak to it" is the scarce signal, and weighting
it below recency would rebuild a generic news feed.

### `state.py`
**Owns** everything persisted. Append-only JSONL; a truncated final line is
skipped rather than fatal.
**Decision:** JSONL in git, not SQLite. The state *is* the product — reviewable in
a diff and readable from a phone — and at this volume a binary blob buys nothing.

### `notify.py`
**Owns** delivery. `Channel.send(payload) -> bool`; channels that raise are caught.
**Decision:** the payload is built by parsing the *committed brief*, not from
in-memory state, so what arrives on the phone and what is in the repo cannot
drift. Python-side channels (Telegram) send directly; session-side channels
(push, email, file) are relayed by the slash command from
`state/last_notification.json`, because Python cannot call them.

### `/radar` and `/video` (Claude)
**Own** editorial judgement and drafting. Deliberately not Python: the decision
"does he have standing to write this, and what's the counter-position" is the
whole value, and it is not expressible as arithmetic.

## External dependencies

| Dependency | Used for | Auth | Failure mode |
|---|---|---|---|
| RSS/Atom feeds (~25) | Most items | none | Per-source, recorded in the report |
| HN Algolia API | Engagement signal | none | Same |
| arXiv API | Research lane | none | Same |
| Telegram Bot API (v1) | Delivery | bot token | Channel reports unavailable, run continues |
| `brandenmillward.github.io` | Duplicate check | none | Claude notes it couldn't check |
