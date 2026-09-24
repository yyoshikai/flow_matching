
# Repository agent instructions

## Working method

- Inspect only the files needed for the task.
- Make the smallest coherent change and avoid unrelated refactors.
- Apply fail-first and never-fallback behavior to all validation and runtime logic.
- Keep common safety behavior in tested Python modules; keep shell adapters thin.
- Do not execute Docker, Slurm, Apptainer, GitHub mutation, or credential operations unless the task explicitly authorizes the exact action.

## Git workflow

- Work only on an authorized `agent/*` branch unless the human explicitly provides another branch. Prefer one branch per workstream for a multi-task milestone; use one branch per task for isolated work.
- The agent may create the authorized agent branch, commit coherent changes, and push normal commits to that branch. Keep task boundaries visible as focused commits and durable reports.
- Never push directly to `work/*`, `dev`, or `main`.
- Never merge a pull request, force-push, rewrite shared history, delete a remote branch, create a tag or release, or change repository settings.
- Keep one active writer per agent branch.
- Stop when the base branch, active workstream/task, branch ownership, or merge target is ambiguous.

## Verification

Run the narrowest relevant test first. Before handoff, run the applicable checks:

```bash
python -m pytest -q
python -m compileall -q src tests
uv run --locked ruff check .
git diff --check
```

Report skipped checks and their reasons.

# Claude Code

- Treat `private_docs/CURRENT_STATE.md` as the portable cross-agent handoff record. Claude auto memory is local convenience, not project evidence or a shared source of truth.
- Use `/research-handoff` only when the human asks for a handoff, checkpoint, or context-transfer summary.
- Use `/research-explain` when a visual or structural explanation would materially reduce reading effort; choose the smallest useful representation.
- Use `research-reviewer` only when the human requests an independent review. Supply the diff and validation evidence; the reviewer has file-reading tools only. Keep implementation and branch ownership in the main session.
