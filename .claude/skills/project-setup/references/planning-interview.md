# Planning Interview

How to get from "I have an idea" to a spec someone could build from without asking you a single follow-up question.

## The specificity standard

This is the single most important thing in the skill. A requirement is only worth writing down if a build could **fail** it.

| Fails the bar | Passes the bar |
|---|---|
| Users can create journal entries | Users create journal entries by first selecting a prompt and then responding to it. Prompts are generated from their past entries. Users can respond by writing or by recording a video of themselves. |
| The app has search | Users search their entries by keyword and by date range; results show the matching entry with the matched phrase highlighted, ranked by recency. |
| It should be fast | A thumbnail generates in under 10 seconds p50; the UI shows progress state the whole time rather than a blocked spinner. |
| Nice, clean UI | Single-screen app: prompt box, generate button, result. No navigation, no settings page. Result appears in place below the input. |

Four things separate the two columns. Check every requirement against them:

1. **Entry point** — how does the user *start* this? ("by first selecting a prompt")
2. **Mechanism** — what actually happens, including where the intelligence comes from? ("prompts are generated from their past entries")
3. **Modality / variants** — what forms does it take? ("write or record a video")
4. **Observable outcome** — how would you know it worked?

When the user gives you a vague requirement, don't just ask "can you be more specific?" — that puts the work back on them and usually produces another vague sentence. Instead **write the specific version yourself** with your best guess and ask them to correct it. "I'm reading that as: [specific version]. Right?" is far more productive, and it's where most of the real design decisions get surfaced.

## Round structure

Aim for 3–5 rounds. Don't run all of these — pick what's actually unknown for this project.

Each round is a short exchange, not a form. The bullets below are what a round *covers*, not a list to fire off at once: put the one or two that genuinely block you into `AskUserQuestion`, and handle the rest by stating your inference and inviting a correction. A round that ends with the user saying "no, it's more like X" has done its job.

### Round 1 — Who and why

The goal is to make the product *for someone*, because "for everyone" produces mush.

- Who is this for? Push for a specific person, not a segment. "YouTubers with under 10k subs who make their own thumbnails and hate it" beats "content creators."
- Is this for you, for other people, or a portfolio piece? This genuinely changes the build — a personal tool doesn't need onboarding, a portfolio piece needs to be demoable in 30 seconds.
- What do they do *today* instead? The status quo is the real competitor, and "nothing, they just don't do it" is a valid and important answer.
- What's the moment of pain that makes someone go looking for this?

### Round 2 — Jobs to be done and the core loop

- Finish this sentence: "When I \_\_\_, I want to \_\_\_, so I can \_\_\_." Get 1–3 of these. These become the jobs-to-be-done section verbatim.
- Walk me through one full use, start to finish, as if narrating a screen recording. This is the highest-yield question in the whole interview — it forces sequence, surfaces screens, and exposes every hand-wave.
- What's the *one* thing that, if it didn't work, would make the whole thing pointless? That's the core function, and it belongs in the MVP alone.
- How often does someone use this — once, daily, in bursts?

### Round 3 — Boundaries

- What is this explicitly *not*? Get at least three. These go straight into "not in scope for now."
- Does it need accounts, payments, or multiple users — and does it need them *now*? For most solo builds the answer is "eventually, not now," and saying so out loud saves an enormous amount of premature architecture.
- What data does it hold, and how bad is it if that leaks? This drives auth, storage, and whether anything can be client-side only.
- Anything it must integrate with? Existing accounts, APIs, a site it has to live on?
- Any hard constraints — budget, deadline, a device it has to run on, a platform it has to ship to?

### Round 4 — Quality bar

Only ask what's load-bearing for this project; skip the rest rather than filling out a form.

- What does "good enough to use" look like versus "good enough to show people"?
- Anything with a speed or cost ceiling? (Especially relevant for anything calling paid model APIs — ask what a single operation is allowed to cost.)
- Does it need to work offline? On mobile? In a specific browser?

## Milestones

Draft the table yourself from the interview, then ask what's wrong with it. Rules that keep it useful:

- **Capabilities, not dates.** "User can download the thumbnail" is a milestone. "Week 3" is not.
- **MVP is brutally small** — single user, one core function, no auth, no history, no batch. If the MVP has more than about four bullets, it isn't an MVP. The test: could this be built in a few sessions and actually used once?
- **Each version is independently useful.** If v1 is worthless without v2, they're one milestone.
- **"Not in scope for now" is a real row**, and it should have contents. It's the row you point at later when a good idea shows up at the wrong time.

Shape:

```
MVP    - Single user. Core function: <the one thing>. <supporting bullet>.
v1     - <capability>, <capability>
v2     - <capability>
Later  - <capability>
Not in scope for now - user accounts, payments, login etc.
```

## Writing the spec document

Use `assets/templates/project_spec.md`. Write it section by section as you go and show each one — a spec revealed all at once gets skimmed and rubber-stamped, and then the errors surface three milestones later.

Two things to check before you call it done:

- **Every requirement passes the specificity bar.** Reread them cold. Any sentence that would survive a completely different implementation needs rewriting.
- **The milestones actually cover the requirements.** Every requirement should map to a version, and every version bullet should trace back to a requirement or a job to be done. Orphans on either side mean something is missing or something is scope creep.
