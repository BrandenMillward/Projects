# {{PROJECT_NAME}} — Project Spec

> Status: draft | Last updated: {{DATE}}
> This is the source of truth for what we're building. Code, architecture, and docs derive from it.

---

## 1. Project Requirements

### Purpose

<!-- What this is, in two or three sentences. Written so someone outside the project understands it. -->

### Who it's for

<!-- A specific person, not a segment. What they do today instead of using this. -->

### Problem it solves

<!-- The moment of pain. Why the status quo is not good enough. -->

### What it does

<!-- The core loop, concretely.

Every requirement below must name the entry point, the mechanism, the modalities, and an
observable outcome. If a sentence would survive a completely different implementation,
it is not specific enough yet.

Bad:  Users can create journal entries
Good: Users create journal entries by first selecting a prompt and then responding to it.
      Prompts are generated based on their past journal entries.
      Users can both write entries or take a video of themselves.
-->

- 
- 
- 

### Jobs to be done

<!-- When I ___, I want to ___, so I can ___. One to three. -->

1. 
2. 

---

## 2. Milestones

Capability-based, not date-based. Each version is independently useful.

| Version | Core app functionality |
|---|---|
| **MVP** | • Single user<br>• **Core function:** <the one thing that must work><br>• <supporting capability> |
| **v1** | • **<Capability>:** <what the user can now do><br>• **<Capability>:** <what the user can now do> |
| **v2** | • **<Capability>:** <what the user can now do> |
| **Later** | • **<Capability>:** <what the user can now do> |
| **Not in scope for now** | User accounts, payments, login etc. |

---

## 3. Engineering Design

**Definition:** technical requirements for how the project will be built.

### Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | | |
| Frontend | | |
| Backend | | |
| Database | | |
| Hosting | | |
| AI models | | |

### Engineering requirements

<!-- Non-functional bar, in numbers where possible: performance, cost per operation,
     data handling and privacy, auth model, rate limits, third-party failure behaviour. -->

- 
- 

### Quality attributes

<!-- Derived from the ISO/IEC 25010:2023 quality model — used here as a checklist against
     forgetting a whole dimension, not as a compliance exercise. Most rows are legitimately
     "not a concern for this project"; say so rather than deleting the row, so the next
     person knows it was considered. Every row that IS a concern needs a number.
     Full targets and the toolchain live in docs/STANDARDS.md. -->

| Characteristic | Target |
|---|---|
| Functional suitability | |
| Performance efficiency | |
| Reliability | |
| Security | |
| Maintainability | |
| Interaction capability | |
| Compatibility / flexibility / safety | |

### Standards

**Coding standard:** see `docs/STANDARDS.md`
**Formal standards regime:** <!-- "none applies" for most projects. If one does, name it
                                  and the clause, and point at where the evidence lives. -->

### Architecture

#### System overview

<!-- Components, what each owns, and one real user action traced end to end. -->

#### Component architecture

<!-- Per component: responsibility, interface, dependencies, decisions that are costly to reverse. -->

### System design

<!-- Data model sketch, external API contracts, where state lives, secrets handling, what runs where. -->

---

## Open questions

<!-- Anything deliberately unresolved, and when it needs an answer. -->

- 
