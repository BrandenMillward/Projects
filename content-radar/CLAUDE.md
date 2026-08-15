# Content Radar

Weekly radar that polls ~29 AI/tech/finance sources and proposes 3–5 blog topics
Branden specifically has standing to write about, then drafts YouTube + TikTok
explainer scripts for the ones he picks. Solo tool, local, no hosted service.

Feeds the existing `blog` skill, which drafts posts for
`brandenmillward.github.io` — this project is the missing front-end to it, not a
replacement for it.

## The split that matters

**Python is deterministic. Claude does the judgement.** `radar fetch` / `score`
only get from ~400 raw items to a shortlist of ~30; deciding which of those are
worth writing, and what the angle is, happens in `/radar`. Do not move editorial
logic into Python, and do not add an Anthropic SDK dependency — the judgement
layer *is* the session, which is why this needs no API key.

## Commands

```bash
uv run pytest                          # 59 tests, all offline
uv run radar fetch --since 7           # poll sources -> state/items.jsonl
uv run radar score --since 7 --top 30  # cluster + score -> state/candidates.json
uv run radar notify --brief briefs/YYYY-MM-DD.md
uv run radar seen                      # mark surfaced candidates as seen
uv run radar prune                     # trim items >45d, seen >180d
uv run ruff check .
```

Slash commands: `/radar` (weekly brief), `/video <n>` (script package).

## Architecture

`sources.py` fetches → `cluster.py` collapses duplicates → `score.py` ranks →
`/radar` judges and writes `briefs/` → `notify.py` delivers. State is
append-only JSONL in `state/`, committed to git on purpose: it is diffable,
greppable and readable on GitHub from a phone. Details in `docs/ARCHITECTURE.md`.

## Constraints

- **Never invent a fact, statistic, quote, client or anecdote.** Every claim in a
  brief or script traces to a fetched source URL. Angles are ours; facts are not.
- **British English throughout, no exceptions** — this feeds published work.
- **No hype vocabulary:** never "game-changing", "revolutionary", "unlock the
  power of", "in the age of AI", "let's dive in".
- **Feeds and public read-only APIs only.** No HTML scraping of article bodies,
  no paywall circumvention, always a descriptive User-Agent.
- **Reddit is not a source** — unauthenticated `.json` returns 403 since May 2026
  and OAuth is approval-gated. Don't add it back without checking.
- **No secrets committed.** `.env` is gitignored. MVP needs no keys at all;
  Telegram (v1) is the only thing that ever adds one.
- Delivery matters as much as generation: a brief that isn't pushed is a brief
  that doesn't get read.

## Repository etiquette

Standalone **private** repo. Private on purpose: `briefs/`, `state/decisions.jsonl`
and `drafts/` hold unpublished content plans and the reasons topics were
rejected. Don't make it public without moving that content out first.

Work on `claude/*` branches, conventional commits, no PR unless asked.

## Scheduled runs

`.github/workflows/radar.yml` runs `fetch` → `score` → `prune` every Monday and
commits the shortlist to `state/`. Claude then reads the repo rather than the
open internet, which is why the editorial half doesn't need egress. A degraded
fetch (>30% of sources failing) fails the job deliberately, so GitHub's own
failure email becomes the alert.

## Documentation map

- `docs/project_spec.md` — requirements, milestones, engineering design
- `docs/ARCHITECTURE.md` — components and a traced run
- `docs/PROJECT_STATUS.md` — where we are, what's next
- `docs/CHANGELOG.md` — what changed
- `.claude/skills/video-drafts/SKILL.md` — video format and voice rules
