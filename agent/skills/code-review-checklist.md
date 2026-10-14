# Code Review Checklist Skill

Use this skill when reviewing a code change before approval.

## Checklist

- The change matches the stated task and adds no unrelated edits.
- New behavior has a test; existing tests still pass.
- Error paths return clear errors instead of panicking.
- No secrets, tokens, or local-only paths are committed.
- Public function behavior stays unchanged unless the task requires it.

## Process

1. Read the full diff, not just the description.
2. Check each new dependency and flag unneeded ones.
3. Run the relevant tests and linters locally.
4. Leave one actionable comment per issue found.
5. Approve only when every item above is satisfied.
