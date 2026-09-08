# Agent Guidelines for ROMUtil

## Design Documentation (`DESIGN.md`) Policy
- **Always update `DESIGN.md`**: Whenever you add, modify, refactor, or delete modules, classes, algorithms, or CLI flags, you MUST update [`DESIGN.md`](./DESIGN.md) to keep documentation in sync with code.
- **Pre-commit validation**: A pre-commit hook runs [`scripts/sync_design_doc.py`](./scripts/sync_design_doc.py) on every commit. It verifies that all modules in `romutil/` are documented, all internal links are valid, and the package tree matches the repository.
- **Tests**: Always run `uv run pytest` before committing. All tests must pass.
