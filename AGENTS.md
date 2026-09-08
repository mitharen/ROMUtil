# Agent Guidelines for ROMUtil

## Design Documentation (`DESIGN.md`) Policy
- **Architectural Focus (Evergreen Design, Not a Changelog)**:
  - [`DESIGN.md`](./DESIGN.md) is an **evergreen system architecture and design document**, NOT a pull request summary, commit log, or changelog of improvements.
  - Always describe what the system *is* and *how it functions*. Never use diff/changelog language such as "recently added", "replaced", "modernized", or "refactored".
  - Focus on **system architecture, algorithmic invariants, mathematical formulations, data flow, and core domain abstractions**.
  - Do NOT document low-level implementation details, syntax choices, or internal refactors (e.g. exhaustive dataclass attribute inventories, type annotations, or Python deprecation workarounds) as design-level topics.
  - Do NOT include volatile or transient metrics (e.g. hardcoded test counts or coverage percentages) in documentation.
- **Pre-commit validation**: A pre-commit hook runs [`scripts/sync_design_doc.py`](./scripts/sync_design_doc.py) on every commit. It verifies that all modules in `romutil/` are documented, all internal links are valid, and the package tree matches the repository.
- **Tests & Coverage Acceptance Criteria**:
  - Always run `uv run pytest` before committing.
  - All tests must pass, and total test coverage across `romutil/` must remain at or above **95%** (enforced automatically by `pytest --cov-fail-under=95`).
  - Any new feature, edge case, or bug fix must be accompanied by corresponding positive and negative test cases.

## Git Workspace & Subagent Concurrency Policy
- **Workspace Isolation Mode**: Always invoke concurrent subagents with `Workspace: 'share'`. This utilizes `git worktree`, sharing the underlying `.git` object store while providing independent working trees and indices.
- **Dedicated Feature Branches**: Every subagent MUST operate on its own unique feature branch (e.g., `feature/<task-name>`). Never commit directly to `main`.
- **Pre-Completion Checks**: Before completing a task, subagents must run `uv run pytest` (enforcing >= 95% coverage) and verify [`scripts/sync_design_doc.py`](./scripts/sync_design_doc.py) passes.
- **No Direct Merges**: Subagents must not merge their own feature branch into `main`. Subagents report their branch name and diff summary back to the parent coordinator for review and integration.
