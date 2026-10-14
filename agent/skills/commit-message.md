# Commit Message Skill

Use this skill when writing a git commit message.

## Rules

- Keep the subject line short, under 72 characters.
- Write the subject in the imperative mood, such as "Fix" or "Add".
- Do not add prefixes like `feat:` unless the project asks for them.
- Leave the body out for small changes; explain only the why for larger ones.
- Wrap body lines at 72 columns.

## Process

1. Run `git status` and `git diff` to see what changed.
2. Stage only the files related to the change.
3. Draft a subject that says what the commit does, not how.
4. Mention the failing test or error the commit addresses, if any.
5. Verify the final diff before committing.
