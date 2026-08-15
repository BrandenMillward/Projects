---
description: Run the weekly content radar — fetch, score, judge, write and deliver the brief.
---

# /radar

Produce this week's content brief. Python does the deterministic work; **you do
the editorial judgement**, which is the only reason this is worth running.

`$ARGUMENTS` may be:
- empty — run the full weekly brief
- `pick <n>` — record topic *n* from the latest brief as picked
- `kill <n> <reason>` — record it as killed (suppressed 90 days)
- `shipped <n>` — record it as published (suppressed permanently)
- `blog <n>` — hand topic *n* to the `blog` skill

## Full run

**1. Fetch and score.**

```bash
cd content-radar
uv run radar fetch --since 7
uv run radar score --since 7 --top 30
```

If `fetch` exits non-zero it is *degraded* — more than 30% of sources failed.
Do not write a brief from a degraded run. Send the failure notification instead
and stop:

```bash
uv run radar notify --brief briefs/$(date +%F).md --failure "N of M sources failed"
```

**2. Judge.** Read `state/candidates.json`. The `score` field got these onto the
shortlist; it does not decide what goes in the brief. For each candidate ask:

- **Does it intersect what Branden actually does?** Multi-agent orchestration,
  agent networks, guardrails, evaluation, explainable AI in regulated
  industries, AI upskilling. If the honest answer is "he'd be commentating, not
  reporting from inside it", drop it.
- **Is there an angle, or would this just restate the consensus?** A topic with
  no counter-position is not a post. Name the prevailing take and what he'd say
  against it.
- **Has he covered it?** Check `suppressed` in `candidates.json`, and fetch
  `https://brandenmillward.github.io/blog/index.html` to check the live archive.
- **Which lane?** AI and regulated-industries topics default to `blog`.
  Generational-finance topics (student loans, pensions, cost of living) default
  to `video` — that audience is on TikTok/Shorts, not on a practitioner blog.
  A finance topic strong enough for both can be marked `both`.

Pick **3–5**. Fewer than 3 means a thin week — send the failure notification
rather than padding the brief.

**3. Write** `briefs/YYYY-MM-DD.md`. The format is parsed by `radar notify`, so
keep the headings and bold labels exactly as shown:

```markdown
# Content Radar — <D Month YYYY>

<N> items across <M> stories. <K> topics worth your time.

## 1. <Title>
**Pitch:** <one sentence — the argument, not the news>
**Why you:** <what gives him standing here specifically>
**Lane:** blog | video | both
**Sources:** <url> · <url>

## Deliberately not suggesting
- **<title>** — <why it was dropped>
```

The "deliberately not suggesting" section is load-bearing: it is what stops the
same rejected theme reappearing week after week, and it shows the filter working.

**4. Deliver and record.**

```bash
uv run radar seen
uv run radar notify --brief briefs/$(date +%F).md
```

Then read `state/last_notification.json` and relay it:
- Call `PushNotification` with the `push` field (one line, already under 200 chars).
- Call `SendUserFile` with the brief path and `status: "proactive"` so it surfaces.

Record each suggestion so future weeks know about it:

```bash
uv run radar decide <cluster_id> "<title>" suggested
```

**5. Commit** the brief and state:
`docs: weekly content brief YYYY-MM-DD`

## Rules

- **Never invent a fact, statistic or quote.** Every claim traces to a fetched
  source URL. Angles are yours; facts are not.
- **British English throughout** — this feeds work published in Branden's voice.
- No hype vocabulary: no "game-changing", "revolutionary", "unlock the power of".
- If a topic needs a personal anecdote to work, say so in the brief rather than
  inventing one.
