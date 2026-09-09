# Agent Guidelines for ROMUtil

## Design Documentation (`DESIGN.md`) Policy
- **Architectural Focus (Evergreen Design, Not a Changelog)**:
  - [`DESIGN.md`](./DESIGN.md) is an **evergreen system architecture and design document**, NOT a pull request summary, commit log, or changelog of improvements.
  - Always describe what the system *is* and *how it functions*. Never use diff/changelog language such as "recently added", "replaced", "modernized", or "refactored".
  - Focus on **system architecture, algorithmic invariants, mathematical formulations, data flow, and core domain abstractions**.
  - Do NOT document low-level implementation details, syntax choices, or internal refactors (e.g. exhaustive dataclass attribute inventories, type annotations, or Python deprecation workarounds) as design-level topics.
  - Do NOT include volatile or transient metrics (e.g. hardcoded test counts or coverage percentages) in documentation.
- **Tests & Coverage Acceptance Criteria**:
  - Always run `uv run pytest` before committing.
  - All tests must pass, and total test coverage across `romutil/` must remain at or above **95%** (enforced automatically by `pytest --cov-fail-under=95`).
  - Any new feature, edge case, or bug fix must be accompanied by corresponding positive and negative test cases.

## Git Workspace & Subagent Concurrency Policy
- **Workspace Isolation Mode**: Always invoke concurrent subagents with `Workspace: 'share'`. This utilizes `git worktree`, sharing the underlying `.git` object store while providing independent working trees and indices.
- **Dedicated Feature Branches**: Every subagent MUST operate on its own unique feature branch (e.g., `feature/<task-name>`). Never commit directly to `main`.
- **Pre-Completion Checks**: Before completing a task, subagents must run `uv run pytest` (enforcing >= 95% coverage).
- **No Direct Merges**: Subagents must not merge their own feature branch into `main`. Subagents report their branch name and diff summary back to the parent coordinator for review and integration.

## Multi-Agent Dispatch Architecture

ROMUtil adheres to a three-tier agent operational model:
1. **Coordinator Agent (Primary Session)**:
   - Orchestrates task decomposition, agent dispatching, and branch integration.
   - Manages the task backlog exclusively in local [`TASKS.md`](./TASKS.md) (untracked via `.gitignore`).
   - Dispatches implementation or optimization tasks to `worker` subagents in isolated shared worktrees (`Workspace: 'share'`).
   - Dispatches `evaluator` subagents to independently audit worker deliverables before merging.
   - Integrates approved feature branches into `main` and updates `TASKS.md`.
2. **Worker Agents (`worker`)**:
   - Build production code, write comprehensive tests, or investigate optimization strategies.
   - Operate on dedicated feature branches (`feature/<task-name>`) in isolated shared worktrees.
   - Maintain $\ge 95\%$ test coverage across `romutil/`.
   - Never commit task roadmaps, backlog lists, or intermediate research drafts into repository files. Return findings directly to the coordinator.
   - Report branch name, commit SHA, diff summary, and intermediate data back to the coordinator.
3. **Evaluator Agents (`evaluator`)**:
   - Independently audit worker branch diffs, code quality, test assertions, and repository hygiene.
   - Run automated validations: `uv run pre-commit run --all-files` and `uv run pytest` ($\ge 95\%$ coverage).
   - Verify that no task descriptions, roadmaps, or scratch notes are leaked into public docs or code.
   - Issue a definitive verdict: `RECOMMEND_MERGE` or `RECOMMEND_REJECT` (with actionable remediation steps).
   - Recommend follow-up tasks and newly revealed edge cases for the coordinator to add to `TASKS.md`.
