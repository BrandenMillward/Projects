---
name: video-drafts
description: Draft explainer video scripts for Branden's YouTube and TikTok/Shorts channels — one research spine cut two ways, in his voice. Use when drafting a video script, a Shorts/TikTok cut, a video hook, or when /video is invoked.
---

# Video drafts

One topic, one argument, **two cuts**. The YouTube explainer and the vertical
short are the same thesis at two lengths — not two separate pieces of work. If
the short says something the long version doesn't argue, one of them is wrong.

Audience differs by cut, and that is the whole reason the format splits:

- **YouTube (5–8 min)** — senior data/AI practitioners and engineering leaders.
  The same people who read the blog. They will tolerate nuance and want the
  caveat.
- **TikTok / Shorts (45–60s)** — a younger, broader audience. This is where
  generational-finance topics (student loans, pensions, cost of living) belong,
  and where a practitioner credential is worth stating once rather than assumed.

## Voice

Inherited from the blog skill, because it is the same person talking.

- **British English throughout. No exceptions.** organisation, utilise, recognise,
  analyse, prioritisation.
- First person, measured, plain-spoken. "In my experience", "one way to look at
  it", "I think that's wrong".
- Open on a concrete moment or the real problem — never a definition, never "In
  today's rapidly evolving landscape".
- **Never:** "game-changing", "revolutionary", "unlock the power of", "in the age
  of AI", "let's dive in", "buckle up". No emoji in the script body. No
  exclamation marks in voiceover.
- Positioning is practitioner-who-ships, not commentator. Where a claim rests on
  his own work, say so plainly; where it rests on a source, cite the source.
- **Never invent an anecdote, client, metric or quote.** If a beat needs one,
  write a defensible placeholder and flag it at the top of the file.

## Output format

Write `drafts/<slug>-video.md` with exactly these two parts.

### Part 1 — YouTube explainer (5–8 min, 900–1,200 words of voiceover)

```markdown
# <Topic> — video package

> **Needs your input:** <any placeholder anecdote to replace, or "none">

## YouTube explainer (~N min)

**Titles** (pick one)
1. <title — under 60 chars, no clickbait, states the claim>
2. <title>
3. <title>

**Thumbnail brief:** <what is in frame, 2–3 words of overlay text max, and the
palette: bg #0b0f14, accent #2dd4bf, cards #151c26 — matches the site>

**Hook (0:00–0:15)**
VO: <the problem or a concrete moment. No channel intro, no "hey everyone".>
On screen: <3–6 words>

**Beat 1 — <label> (0:15–1:30)**
VO: <...>
On screen: <...>
B-roll: <a prompt specific enough to generate or source from — name the subject,
the composition and the mood, not just "AI imagery">

<repeat for 4–6 beats total>

**Close (last 30s)**
VO: <the takeaway, then a short forward-looking line. One CTA maximum.>

**Description:** <2–3 sentences, primary keyword in the first line>
**Chapters:** 0:00 <label> / 0:15 <label> / ...
**Tags:** <8–12, comma separated>
```

### Part 2 — TikTok / Shorts vertical (45–60s, 130–160 words)

```markdown
## Vertical cut (45–60s)

**Hook (0:00–0:03)** — must land inside three seconds or the rest never plays.
VO: <one sentence, the sharpest claim in the whole piece>
On screen: <4–8 words, large, upper third — the middle is where the UI sits>

**Beat 1 (0:03–0:20)** / **Beat 2 (0:20–0:40)** / **Beat 3 (0:40–0:60)**
VO: <...>
On screen: <...>
B-roll: <...>

**Caption:** <under 150 chars, front-loads the claim>
**Hashtags:** <5–8, mixed broad and niche>
```

## Constraints worth checking before you hand it over

- **Read the hook aloud.** If it takes longer than three seconds, it is too long
  — this is the single highest-leverage line in the short.
- **Voiceover word count implies runtime** at roughly 150 words/minute. A "5
  minute" script of 400 words is not a 5 minute script.
- **On-screen text sits in the upper third** for the vertical cut; platform UI
  covers the lower portion.
- Every factual claim traces to a source from the brief. State the number, not
  "studies show".
- The vertical cut must stand alone. Someone who never sees the long version
  should still get a complete argument.
- Run the British-English check over both cuts before finishing.
