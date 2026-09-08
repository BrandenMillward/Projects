---
description: Open a change record from a one-line request, at the specificity bar
allowed-tools: Bash(python3*), Bash(git status*), Bash(git log*), Read, Grep, Glob, Edit, Write
---

Open a change record for: $ARGUMENTS

1. **Check the threshold first.** If this is a bug fix inside existing behaviour, a refactor with no interface change, or a patch bump, say so in one line and do the work — do not open a record. The threshold rule is in the skill's Phase 0.

2. **Write the request at the bar.** What changes, what it replaces, what stays the same. If `$ARGUMENTS` is vague, write the specific version yourself and ask the user to correct it rather than asking them to be more specific.

3. **Classify** it: `feature`, `change`, `deprecation`, `migration`, `dependency`, `security`, or `fix`.

4. **Create the record:**

```bash
python3 <skill-dir>/scripts/changes.py new "<title>" --type <type> --dir .
```

5. **Fill the Request section** from step 2, including what it traces to. Leave Impact, Decision, and Verification empty — those are later phases.

6. Show the record path and the request text back. Ask whether to run the impact assessment now (`/impact <id>`) or stop here.

Do not assess impact, amend the spec, or write code in this command. Opening a record is its own step.
