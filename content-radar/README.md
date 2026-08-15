# Content Radar

A weekly radar that reads the AI, tech and finance internet so Branden doesn't
have to, and proposes 3–5 topics he specifically has standing to write about —
then drafts YouTube and TikTok explainer scripts for the ones he picks.

It is the missing front-end to the existing `blog` skill, which drafts posts for
[brandenmillward.github.io](https://brandenmillward.github.io) but starts cold
every run with no memory of what was suggested, picked or rejected before.

## How it works

```
~29 sources  ->  cluster  ->  score  ->  Claude judges  ->  brief  ->  delivered
  RSS/HN/arXiv    dedupe      heuristic   the editorial     3-5       push, email,
                  by URL      shortlist   layer            topics     Telegram
                  + title     of ~30
```

**Python is deterministic; Claude does the judgement.** The scoring heuristic
only gets from ~400 raw items to a shortlist of ~30. Deciding which of those are
actually worth writing — does this intersect what he does, is there a
counter-position, has he covered it already — happens in the `/radar` slash
command. That split is the main design decision: it keeps the Python
unit-testable and means the project needs no API key at all.

## Quick start

```bash
git clone https://github.com/BrandenMillward/news-scraper-video-draft.git
cd news-scraper-video-draft
uv venv && uv pip install -e ".[dev]"
uv run pytest                          # 59 tests, all offline

uv run radar fetch --since 7           # poll sources
uv run radar score --since 7 --top 30  # cluster + rank
uv run radar prune                     # keep committed state bounded
```

Then run `/radar` in Claude Code to produce and deliver the brief, and
`/video <n>` to draft the script package for a topic.

## What gets delivered

The brief is **pushed, not parked**. Committing markdown to `briefs/` is the
archive; delivery is a phone push, an email carrying the full topic list (so the
week can be judged without opening anything), the brief file surfaced in-session,
and optionally Telegram. A run that crashes, loses more than 30% of its sources,
or finds fewer than three topics notifies too — a silently broken radar and a
quiet news week must never look the same.

## Layout

| Path | What |
|---|---|
| `radar/` | The pipeline — fetch, cluster, score, state, notify |
| `radar/feeds.toml` | Source list; adding one is a single entry |
| `state/` | Append-only JSONL, committed on purpose: diffable and phone-readable |
| `briefs/` | One markdown brief per week |
| `drafts/` | Video script packages |
| `.claude/commands/` | `/radar`, `/video` |
| `docs/` | Spec, architecture, status, changelog |

## Where it runs

`.github/workflows/radar.yml` does the deterministic half on a schedule —
Monday 06:07 UTC, `fetch` → `score` → `prune`, committing the shortlist. GitHub
Actions has open egress and runs whether or not your laptop is on. The editorial
half then reads this repo rather than the internet.

Run it by hand any time from the Actions tab (**Run workflow**), or locally with
the commands above.

This repo is **private** deliberately: `briefs/`, `drafts/` and the decision log
hold unpublished content plans and the reasons topics were rejected.

## Status

MVP is built and verified offline — **59 tests**; fixture runs confirm duplicate
collapse across four outlets, signal ranking 6× above noise, week-two demotion
from 0.84 to 0.13, and the failure paths.

**`radar fetch` has not yet run against live sources.** The environment it was
built in blocks outbound egress to feed hosts, so some of the ~29 feed URLs will
be wrong. The first Actions run — or a local run — will name exactly which. The
per-source failure report exists to make that a quick fix. See
`docs/PROJECT_STATUS.md`.
