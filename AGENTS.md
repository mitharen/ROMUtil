# Agent Guidelines for ROMUtil

## Design Documentation (`docs/DESIGN.md`) Policy
- **Architectural Focus (Evergreen Design, Not a Changelog)**:
  - [`docs/DESIGN.md`](./docs/DESIGN.md) is an **evergreen system architecture and design document**, NOT a pull request summary, commit log, or changelog of improvements.
  - Always describe what the system *is* and *how it functions*. Never use diff/changelog language such as "recently added", "replaced", "modernized", or "refactored".
  - Focus on **system architecture, algorithmic invariants, mathematical formulations, data flow, and core domain abstractions**.
  - Do NOT document low-level implementation details, syntax choices, or internal refactors (e.g. exhaustive dataclass attribute inventories, type annotations, or Python deprecation workarounds) as design-level topics.
  - Do NOT include volatile or transient metrics (e.g. hardcoded test counts or coverage percentages) in documentation.
- **Tests & Coverage Acceptance Criteria**:
  - Always run `uv run pytest` before committing.
  - All tests must pass, and total test coverage across `romutil/` must remain at or above **95%** (enforced automatically by `pytest --cov-fail-under=95`).
  - Any new feature, edge case, or bug fix must be accompanied by corresponding positive and negative test cases.

## Git & Branching Policy
- **Dedicated Feature Branches**: Every agent or contributor must operate on its own unique feature branch (e.g., `feature/<task-name>`). Never commit directly to `main`.
- **Pre-Completion Checks**: Before completing a task or submitting a pull request, run `uv run pre-commit run --all-files` and `uv run pytest` (enforcing >= 95% coverage).
- **No Unreviewed Direct Merges**: Feature branches must be verified through clean tests and code review before integration into `main`.

## Local Example Artifact Generation Policy
- **Visual Impact Verification on Task Completion**:
  - Whenever a task affecting parsing, layout, or rendering is merged into `main`, the coordinator must automatically produce local example outputs to enable visual inspection of the changes.
  - Build local map outputs using `uv run python scripts/build_pages.py --solver-timeout 10` (or targeted `romutil` CLI invocations) outputting to `_site/`.
  - Provide direct clickable `file://` links to key outputs (e.g., `[Site Index](file:///home/user/proj/ROMUtil/_site/index.html)`, `[Midgaard HTML](file:///home/user/proj/ROMUtil/_site/midgaard.html)`, `[Midgaard Metropolitan HTML](file:///home/user/proj/ROMUtil/_site/midgaard_metropolitan.html)`) in the completion summary for user review.
