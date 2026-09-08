---
id: {{ID}}
title: {{TITLE}}
type: {{TYPE}}
status: {{STATUS}}
reversibility: {{REVERSIBILITY}}
retroactive: false
opened: {{DATE}}
decided_by:
decided_on:
shipped:
traces_to:
---

# {{ID}} — {{TITLE}}

## Request

<!-- One paragraph, at the specificity bar: what changes, what it replaces, what stays the same.
     A finished implementation must be able to FAIL this. If it would survive any implementation,
     rewrite it before going further. -->

**What changes:**

**What it replaces:**

**What stays the same:**

**Traces to:** <!-- requirement, job to be done, or decision this serves. If nothing: say so, and ask why. -->

## Impact

<!-- Fill only the surfaces that are actually touched. Delete the rest — empty sections are padding. -->

| Surface | Impact |
|---|---|
| Spec | |
| Code | |
| Data | |
| Config / cost | |
| Standards | |
| Docs | |

### Spec amendment

<!-- Quote the current requirement text, then the amended version. The diff is the deliverable. -->

**Currently:**

**Becomes:**

## Reversibility

**Rating:** {{REVERSIBILITY}} <!-- cheap | annoying | one-way-door -->

**Why:**

**Rollback path:** <!-- Actual steps. "Restore from backup" is not a rollback path.
                       For annoying and one-way-door: has this been tried? -->

## Decision

**Outcome:** <!-- approved | approved-with-conditions | deferred | rejected -->
**Decided by:**
**Date:**
**Reasoning:**

**Conditions:** <!-- If approved with conditions, they become verification criteria below. -->

## Standards

```
Code gate      <!-- pass/fail — name the commands that ran -->
Standards      <!-- n/a, or the clause and the evidence -->
Quality        <!-- the measurement, against the target -->
```

## Verification

<!-- How we know it worked. Evidence, not assertion. Unverified is not shipped. -->

- [ ] 
- [ ] 

## Follow-ups

<!-- Actions arising that are NOT part of this change. Each becomes its own request. -->

- 
