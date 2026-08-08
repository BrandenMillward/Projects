---
description: Draft YouTube + TikTok explainer scripts for a topic from the latest brief.
---

# /video

Turn topic `$ARGUMENTS` (a number from the latest brief, or a topic in your own
words) into a two-cut script package.

Load the `video-drafts` skill in `.claude/skills/video-drafts/SKILL.md` and
follow it — it carries the format, the timing constraints and the voice rules.

## Before drafting

1. Read the latest `briefs/*.md` and take the topic's angle, why-you and
   sources. **Do not re-research from scratch** — the brief already did that,
   and the video must argue the same thing the brief proposed.
2. If any factual claim is time-sensitive, verify it with WebSearch/WebFetch
   this run rather than trusting the brief's snapshot.
3. If the strongest hook needs a specific memory or client story Branden hasn't
   given you, write a defensible version and flag it at the top of the file as
   the one line worth replacing. Never invent the anecdote.

## Output

Write `drafts/<slug>-video.md` containing **both cuts of one research spine** —
the YouTube explainer and the vertical short are the same argument at two
lengths, not two separate pieces.

Then record the decision and tell him it is ready:

```bash
cd content-radar && uv run radar decide <cluster_id> "<title>" picked --reason "video drafted"
```

Call `SendUserFile` with the draft path so it surfaces rather than sitting in
the repo.
