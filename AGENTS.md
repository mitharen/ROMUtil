# Agent Guidelines for ROMUtil

## Design Documentation (`DESIGN.md`) Policy
- **Always update `DESIGN.md`**: Whenever you add, modify, refactor, or delete modules, classes, algorithms, or CLI flags, you MUST update [`DESIGN.md`](./DESIGN.md) to keep documentation in sync with code.
- **Pre-commit validation**: A pre-commit hook runs [`scripts/sync_design_doc.py`](./scripts/sync_design_doc.py) on every commit. It verifies that all modules in `romutil/` are documented, all internal links are valid, and the package tree matches the repository.
- **Tests & Coverage Acceptance Criteria**:
  - Always run `uv run pytest` before committing.
  - All tests must pass, and total test coverage across `romutil/` must remain at or above **95%** (enforced automatically by `pytest --cov-fail-under=95`).
  - Any new feature, edge case, or bug fix must be accompanied by corresponding positive and negative test cases.

## Git Workspace & Subagent Concurrency Policy
- **Workspace Isolation Mode**: Always invoke concurrent subagents with `Workspace: 'share'`. This utilizes `git worktree`, sharing the underlying `.git` object store while providing independent working trees and indices.
- **Dedicated Feature Branches**: Every subagent MUST operate on its own unique feature branch (e.g., `feature/<task-name>`). Never commit directly to `master`.
- **Pre-Completion Checks**: Before completing a task, subagents must run `uv run pytest` (enforcing >= 95% coverage) and verify [`scripts/sync_design_doc.py`](./scripts/sync_design_doc.py) passes.
- **No Direct Merges**: Subagents must not merge their own feature branch into `master`. Subagents report their branch name and diff summary back to the parent coordinator for review and integration.
