# Engineering Design

**Definition:** the technical requirements for how the project will be built.
**Covers:** tech stack, engineering requirements, architecture, system design.

The output is the third section of `project_spec.md`. Getting here requires a research step first, because stack advice from memory is stale advice — model names, pricing, free-tier limits, and framework major versions all move faster than training data.

## Step 1 — Research report

Produce a short report *before* asking the user to choose anything. The point is to make the decision informed, not to show off breadth.

**Look things up rather than recalling them.** Use `WebSearch` / `WebFetch` for anything with a number attached: model pricing and capabilities, free-tier limits, current major versions, whether a service still has the plan you remember. For anything involving Claude or the Anthropic API, load the `claude-api` skill instead of searching — it's authoritative for model IDs and pricing. If a lookup fails, say the number is unverified rather than guessing.

Scope the research to what this project actually forces a decision on. A static site doesn't need a database comparison. Three to five layers is normal.

Report format — keep it to roughly one screen per layer:

```
## <Layer, e.g. Image generation model>

**What this project needs from it:** <one line, derived from the requirements>

| Option | Fits because | Costs / limits | Watch out for |
|---|---|---|---|
| ... | ... | ... | ... |

**Recommendation:** <option> — <one or two sentences on why, tied to a specific requirement or milestone>
```

Then a closing paragraph: the recommended stack end to end, what it costs to run at MVP scale, and the one decision that would be most expensive to reverse later. That last line matters more than the rest of the report — it tells the user where to spend their attention.

## Step 2 — Ask

Present the choice per layer with `AskUserQuestion`, recommendation first and marked `(Recommended)`. Batch the layers into one call where the answers don't depend on each other.

Don't ask about layers where the project only realistically supports one answer — state the choice and the reason, and move on. Asking someone to pick between three options you've already ruled out two of is theatre.

### House menu

The user's usual candidate set. Treat it as the default shortlist, not a restriction — if something outside it is clearly right, recommend that and say why.

| Layer | Options |
|---|---|
| Language | Python, TypeScript, Swift |
| Frontend | Next.js, Vue.js, Angular |
| Backend | FastAPI, Node/Express, Django |
| Database | MongoDB, Supabase |
| Cloud / VMs | DigitalOcean, Google Cloud, Hetzner, AWS, Azure |
| AI models | Anthropic, OpenAI, Gemini |

Rules of thumb worth stating out loud when they apply:

- **Next.js alone covers a lot.** If the backend is a handful of API routes, a separate backend service is premature. Say so rather than defaulting to a two-service architecture.
- **Supabase bundles Postgres + auth + storage + row-level security.** For anything that will eventually need accounts, it removes a later migration. Mongo is the better fit for genuinely schemaless documents.
- **Vercel is the path of least resistance for Next.js**; a VM (Hetzner/DO) wins when there's a long-running or GPU-ish workload, or when egress costs would bite.
- **Match the AI provider to the modality**, not to brand loyalty — image and video generation, text reasoning, and cheap classification often land on different providers, and using two is fine.
- **Pick the language the user actually writes.** A stack they can debug at midnight beats a marginally better one they can't.

## Step 3 — Write the section

Four subsections in `project_spec.md`:

**Tech stack** — the table of what was chosen and the one-line reason for each. The reasons are the valuable part; six months later nobody remembers why, and the reasons are what tell you whether a decision is still valid.

**Engineering requirements** — the non-functional bar, stated in numbers wherever possible: performance targets, cost ceilings per operation, data handling and privacy, auth model, rate limits, what happens when a third-party API fails. Vague entries here are as useless as vague product requirements; "should be fast" fails the same bar as "users can create entries."

**Architecture** — a system overview: the components, what each is responsible for, and how a request flows through them end to end. Trace one real user action all the way through — that's what makes it concrete instead of a box diagram. A Mermaid diagram is worth including if there are more than three components.

**System design** — the parts that need a decision rather than a description: data model sketch, external API contracts, where state lives, how secrets are handled, what runs where.

Keep this section at the level of *decisions and interfaces*. Implementation detail belongs in `docs/ARCHITECTURE.md`, which gets expanded during the build; the spec is what the architecture doc is generated from.
