---
description: Assess blast radius and reversibility for an open change record
allowed-tools: Bash(python3*), Bash(git*), Read, Grep, Glob, Edit, Task
---

Run the impact assessment for change: $ARGUMENTS

1. **Read the record** (`python3 <skill-dir>/scripts/changes.py show <id> --dir .`) and the project spec. Never re-ask what the record already answers.

2. **Dispatch the `impact-analyzer` subagent** with the request text and the repo root. It reads broadly; only its report should come back. Do it inline only if the change is obviously contained to one file already open.

3. **Walk the six surfaces** — spec, code, data, config/cost, standards, docs. Stop early on any surface that comes back empty and delete it from the table rather than writing "no impact".

4. **Rate reversibility** — cheap / annoying / one-way door. Rate the consequence, not the diff size. If the rating is one-way door, propose the cheaper version before accepting it: expand-then-contract, flag it, copy don't move, soft delete, or test the reverse first.

5. **Write the assessment into the record**, and set the reversibility field:

```bash
python3 <skill-dir>/scripts/changes.py set <id> --reversibility <tier> --dir .
```

6. **Close with a recommendation** — approve, approve a reduced version, defer, or reject — and the reason. An assessment that refuses to recommend has left the hardest part undone.

Then present the assessment and the decision as one `AskUserQuestion` call. Do not start implementation.
