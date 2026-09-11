#!/usr/bin/env python3
"""
validate_corpus.py - Bulk External Corpus Validation Harness for ROMUtil.

Stress-tests ROMUtil's area parser against authoritative, public MUD area
repositories cataloged in docs/MUD_REPOSITORIES.md without bloating version
control history.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from romutil.models import AreaData
from romutil.parser import Parser

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("validate_corpus")


@dataclass(frozen=True)
class RepoEntry:
    """Authoritative external MUD repository descriptor."""

    name: str
    slug: str
    url: str
    branch: str
    commit: str
    area_path: str
    dialect: str


@dataclass
class FileFailure:
    """Details of a single failed area file parse."""

    file: str
    error: str
    traceback: str


@dataclass
class RepoResult:
    """Validation metrics and status for a single repository."""

    name: str
    slug: str
    dialect: str
    commit: str
    area_path: str
    status: str  # 'success', 'failed', 'error', 'dry-run'
    files_scanned: int = 0
    files_parsed: int = 0
    files_failed: int = 0
    rooms: int = 0
    exits: int = 0
    failures: list[FileFailure] = field(default_factory=list)
    error_message: str | None = None


# Authoritative catalog mirroring docs/MUD_REPOSITORIES.md (Section 3)
REPOSITORY_REGISTRY: list[RepoEntry] = [
    RepoEntry(
        name="QuickMUD",
        slug="avinson/rom24-quickmud",
        url="https://github.com/avinson/rom24-quickmud",
        branch="master",
        commit="364c26f1b124e238156e3d11b4e72a8992c66b74",
        area_path="area",
        dialect="rom",
    ),
    RepoEntry(
        name="ROM 2.4b6",
        slug="DikuMUDOmnibus/ROM",
        url="https://github.com/DikuMUDOmnibus/ROM",
        branch="master",
        commit="f03fc881e1b7c9e06465bc61d86d28317fd52a6b",
        area_path="area",
        dialect="rom",
    ),
    RepoEntry(
        name="RaM-Fire",
        slug="DikuMUDOmnibus/RaM-Fire",
        url="https://github.com/DikuMUDOmnibus/RaM-Fire",
        branch="master",
        commit="9ad7a9dfe50eb8d8ca1f60be08c70fe23b965fd2",
        area_path="Fire/area",
        dialect="rom",
    ),
    RepoEntry(
        name="Merc 2.1",
        slug="alexmchale/merc-mud",
        url="https://github.com/alexmchale/merc-mud",
        branch="master",
        commit="358bed459176ce54ef5560ab52e5939d683ab21a",
        area_path="area",
        dialect="merc",
    ),
    RepoEntry(
        name="Merc 2.2",
        slug="iam-TJ/merc",
        url="https://github.com/iam-TJ/merc",
        branch="master",
        commit="51320cd4b5eff3143e99381ba68966d73eb99d4f",
        area_path="area",
        dialect="merc",
    ),
    RepoEntry(
        name="EnvyMUD",
        slug="lolindrath/EnvyMUD",
        url="https://github.com/lolindrath/EnvyMUD",
        branch="master",
        commit="0bab3b701ab23247d2788cf37335106adbedb85c",
        area_path="area",
        dialect="envy",
    ),
    RepoEntry(
        name="Ultra-Envy",
        slug="DikuMUDOmnibus/Ultra-Envy",
        url="https://github.com/DikuMUDOmnibus/Ultra-Envy",
        branch="master",
        commit="240666877f5c616bfbf9af803bd9dbade0f48bd3",
        area_path="area",
        dialect="envy",
    ),
    RepoEntry(
        name="DikuMUD Alfa",
        slug="Seifert69/DikuMUD",
        url="https://github.com/Seifert69/DikuMUD",
        branch="master",
        commit="81b74dce0436b782d08b19064e32013c73525b45",
        area_path="dm-dist-alfa/lib",
        dialect="diku",
    ),
    RepoEntry(
        name="CircleMUD 3.1",
        slug="Yuffster/CircleMUD",
        url="https://github.com/Yuffster/CircleMUD",
        branch="master",
        commit="12a0cedbb563465029c46de9537c96854bc11610",
        area_path="lib/world/wld",
        dialect="circlemud",
    ),
    RepoEntry(
        name="tbaMUD",
        slug="tbamud/tbamud",
        url="https://github.com/tbamud/tbamud",
        branch="master",
        commit="bd92753e29f12113ccb5ca9c303927ea168ef640",
        area_path="lib/world/wld",
        dialect="circlemud",
    ),
    RepoEntry(
        name="SmaugFUSS",
        slug="Arthmoor/SmaugFUSS",
        url="https://github.com/Arthmoor/SmaugFUSS",
        branch="master",
        commit="0aff8ad04e12d17084f4aaeaed6d158937d425cd",
        area_path="area",
        dialect="smaug",
    ),
    RepoEntry(
        name="SMAUG Core",
        slug="smaugmuds/_smaug_",
        url="https://github.com/smaugmuds/_smaug_",
        branch="master",
        commit="a13b913cec1e7c8c5732f20b799cb609ebad67f6",
        area_path="db/area",
        dialect="smaug",
    ),
    RepoEntry(
        name="ANATOLIA",
        slug="jaromil/anatoliamud",
        url="https://github.com/jaromil/anatoliamud",
        branch="master",
        commit="2b2ee2cc30c9c983f7ca853f0891e85dc75802d5",
        area_path="lib/areas",
        dialect="anatolia",
    ),
    RepoEntry(
        name="AckFUSS",
        slug="Kline-/ackfuss",
        url="https://github.com/Kline-/ackfuss",
        branch="master",
        commit="6404aa5d00ca13e7e13b90f1934611143ed5d989",
        area_path="area",
        dialect="ackmud",
    ),
    RepoEntry(
        name="AckMUD Classic",
        slug="DikuMUDOmnibus/AckMUD",
        url="https://github.com/DikuMUDOmnibus/AckMUD",
        branch="master",
        commit="a2b2e17373b3b95bb8acce747ea1db17afba7bd3",
        area_path="area",
        dialect="ackmud",
    ),
]


def run_git_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    """Execute a git command with error capturing and path safety."""
    try:
        res = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except FileNotFoundError as exc:
        raise RuntimeError("git executable not found in system PATH") from exc
    except subprocess.CalledProcessError as exc:
        err_msg = exc.stderr.strip() or exc.stdout.strip() or f"Process exited with code {exc.returncode}"
        raise RuntimeError(f"Git command failed ({' '.join(cmd)}): {err_msg}") from exc


def filter_repos(
    repo_target: str = "all",
    dialect_target: str = "all",
    registry: list[RepoEntry] | None = None,
) -> list[RepoEntry]:
    """Filter repositories by name/slug and dialect."""
    entries = registry if registry is not None else REPOSITORY_REGISTRY
    repo_norm = repo_target.strip().lower()
    dialect_norm = dialect_target.strip().lower()

    filtered: list[RepoEntry] = []
    for entry in entries:
        if repo_norm != "all":
            name_norm = entry.name.lower()
            slug_norm = entry.slug.lower()
            slug_leaf = entry.slug.split("/")[-1].lower()
            slug_clean = re.sub(r"[^a-z0-9]", "", slug_norm)
            name_clean = re.sub(r"[^a-z0-9]", "", name_norm)
            query_clean = re.sub(r"[^a-z0-9]", "", repo_norm)

            matches_repo = (
                repo_norm in (name_norm, slug_norm, slug_leaf)
                or (query_clean and query_clean in (name_clean, slug_clean))
            )
            if not matches_repo:
                continue

        if dialect_norm != "all":
            if entry.dialect.lower() != dialect_norm:
                continue

        filtered.append(entry)

    return filtered


def clone_or_checkout_repo(
    entry: RepoEntry,
    cache_dir: Path,
    verbose: bool = False,
) -> Path:
    """Shallow clone or checkout pinned commit SHA into local cache directory."""
    safe_name = entry.slug.replace("/", "_")
    repo_dest = cache_dir / safe_name

    if (repo_dest / ".git").is_dir():
        current_head = run_git_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dest)
        if current_head.startswith(entry.commit) or entry.commit.startswith(current_head):
            if verbose:
                log.info(f"[{entry.name}] Already at commit {entry.commit[:10]}")
            return repo_dest

        # Try local checkout first
        try:
            run_git_cmd(["git", "checkout", entry.commit], cwd=repo_dest)
            return repo_dest
        except RuntimeError:
            pass

        # Fetch commit shallowly from origin
        try:
            run_git_cmd(["git", "fetch", "--depth", "1", "origin", entry.commit], cwd=repo_dest)
            run_git_cmd(["git", "checkout", entry.commit], cwd=repo_dest)
            return repo_dest
        except RuntimeError:
            run_git_cmd(["git", "fetch", "--depth", "50", "origin", entry.branch], cwd=repo_dest)
            run_git_cmd(["git", "checkout", entry.commit], cwd=repo_dest)
            return repo_dest

    # Directory does not exist or has broken .git
    if repo_dest.exists():
        shutil.rmtree(repo_dest)

    cache_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        log.info(f"[{entry.name}] Cloning {entry.url} ({entry.branch})...")

    try:
        run_git_cmd(["git", "clone", "--depth", "1", "--branch", entry.branch, entry.url, str(repo_dest)])
    except RuntimeError:
        run_git_cmd(["git", "clone", entry.url, str(repo_dest)])

    current_head = run_git_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dest)
    if not (current_head.startswith(entry.commit) or entry.commit.startswith(current_head)):
        try:
            run_git_cmd(["git", "fetch", "--depth", "1", "origin", entry.commit], cwd=repo_dest)
            run_git_cmd(["git", "checkout", entry.commit], cwd=repo_dest)
        except RuntimeError:
            run_git_cmd(["git", "fetch", "--depth", "50", "origin", entry.branch], cwd=repo_dest)
            run_git_cmd(["git", "checkout", entry.commit], cwd=repo_dest)

    return repo_dest


def validate_repository(
    entry: RepoEntry,
    cache_dir: Path,
    limit: int | None = None,
    verbose: bool = False,
) -> RepoResult:
    """Validate area files in a repository, tracking room/exit metrics and failures."""
    try:
        repo_path = clone_or_checkout_repo(entry, cache_dir, verbose=verbose)
    except Exception as exc:
        log.error(f"[{entry.name}] Failed to clone/checkout: {exc}")
        return RepoResult(
            name=entry.name,
            slug=entry.slug,
            dialect=entry.dialect,
            commit=entry.commit,
            area_path=entry.area_path,
            status="error",
            error_message=str(exc),
        )

    area_dir = repo_path / entry.area_path
    if not area_dir.is_dir():
        err_msg = f"Configured area directory '{entry.area_path}' does not exist in repository {entry.name}"
        log.error(f"[{entry.name}] {err_msg}")
        return RepoResult(
            name=entry.name,
            slug=entry.slug,
            dialect=entry.dialect,
            commit=entry.commit,
            area_path=entry.area_path,
            status="error",
            error_message=err_msg,
        )

    candidate_files: list[Path] = [
        p
        for p in sorted(area_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in (".are", ".wld")
        and not p.name.endswith("~")
        and not p.name.startswith(".")
    ]

    files_to_parse = candidate_files[:limit] if (limit is not None and limit > 0) else candidate_files

    result = RepoResult(
        name=entry.name,
        slug=entry.slug,
        dialect=entry.dialect,
        commit=entry.commit,
        area_path=entry.area_path,
        status="success",
        files_scanned=len(files_to_parse),
    )

    for file_path in files_to_parse:
        try:
            content = file_path.read_text(encoding="latin-1", errors="replace")
            parser = Parser()
            area_data = parser.parse(content)
            if not isinstance(area_data, AreaData):
                raise ValueError(f"Parser returned invalid result type: {type(area_data).__name__}")

            rooms_count = len(area_data.rooms)
            exits_count = sum(len(r.exits) for r in area_data.rooms)

            result.files_parsed += 1
            result.rooms += rooms_count
            result.exits += exits_count

            if verbose:
                log.info(f"[{entry.name}] OK: {file_path.name} ({rooms_count} rooms, {exits_count} exits)")
        except Exception as exc:
            result.files_failed += 1
            tb_str = traceback.format_exc()
            rel_file = str(file_path.relative_to(repo_path))
            result.failures.append(
                FileFailure(
                    file=rel_file,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=tb_str,
                )
            )
            if verbose:
                log.warning(f"[{entry.name}] FAIL: {file_path.name}: {exc}")

    if result.files_failed > 0:
        result.status = "failed"

    return result


def build_summary_dict(
    results: list[RepoResult],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Construct structured summary dictionary for JSON output."""
    total_scanned = sum(r.files_scanned for r in results)
    total_parsed = sum(r.files_parsed for r in results)
    total_failed = sum(r.files_failed for r in results)
    total_rooms = sum(r.rooms for r in results)
    total_exits = sum(r.exits for r in results)
    overall_success = round(total_parsed / total_scanned, 4) if total_scanned > 0 else 1.0

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "summary": {
            "repositories_processed": len(results),
            "total_files_scanned": total_scanned,
            "total_files_parsed": total_parsed,
            "total_files_failed": total_failed,
            "total_rooms": total_rooms,
            "total_exits": total_exits,
            "overall_success_rate": overall_success,
        },
        "repositories": {
            r.name: {
                "name": r.name,
                "slug": r.slug,
                "dialect": r.dialect,
                "commit": r.commit,
                "area_path": r.area_path,
                "status": r.status,
                "files_scanned": r.files_scanned,
                "files_parsed": r.files_parsed,
                "files_failed": r.files_failed,
                "rooms": r.rooms,
                "exits": r.exits,
                "success_rate": round(r.files_parsed / r.files_scanned, 4) if r.files_scanned > 0 else 1.0,
                "failures": [
                    {"file": f.file, "error": f.error, "traceback": f.traceback}
                    for f in r.failures
                ],
                "error_message": r.error_message,
            }
            for r in results
        },
    }


def print_summary_table(results: list[RepoResult]) -> None:
    """Print formatted ASCII metrics table to stdout."""
    print()
    print("=" * 88)
    print("                 ROMUtil Bulk External Corpus Validation Summary")
    print("=" * 88)
    print(f"{'Repository':<18} {'Dialect':<10} {'Scanned':>7} {'Parsed':>7} {'Failed':>7} {'Rooms':>8} {'Exits':>8} {'Status':<8}")
    print("-" * 88)

    for r in results:
        status_label = r.status.upper()
        print(
            f"{r.name:<18} {r.dialect:<10} {r.files_scanned:>7} {r.files_parsed:>7} "
            f"{r.files_failed:>7} {r.rooms:>8} {r.exits:>8} {status_label:<8}"
        )

    print("-" * 88)
    total_scanned = sum(r.files_scanned for r in results)
    total_parsed = sum(r.files_parsed for r in results)
    total_failed = sum(r.files_failed for r in results)
    total_rooms = sum(r.rooms for r in results)
    total_exits = sum(r.exits for r in results)
    rate = (total_parsed / total_scanned * 100.0) if total_scanned > 0 else 100.0

    print(
        f"Total: {len(results)} repos, {total_scanned} files scanned, {total_parsed} parsed, "
        f"{total_failed} failed ({rate:.1f}% success)"
    )
    print(f"Total Rooms: {total_rooms:,} | Total Exits: {total_exits:,}")
    print("=" * 88)
    print()


def create_argument_parser() -> argparse.ArgumentParser:
    """Configure CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="ROMUtil Bulk External Corpus Validation Harness - Stress-test area parsers against authoritative public MUD repositories.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="all",
        help="Target specific repository name/slug or 'all' (default: all).",
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default="all",
        help="Target specific dialect (e.g. rom, merc, envy, diku, circlemud, smaug, anatolia, ackmud) or 'all' (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max files to parse per repository (optional, integer >= 1).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".corpus_cache"),
        help="Local cache directory for shallow clones (default: .corpus_cache).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write output metrics in JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display planned clone and validation steps without downloading or parsing.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Detailed per-file parse logs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI main entry point. Returns process exit code."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        sys.stderr.write("Error: --limit must be a positive integer >= 1\n")
        return 2

    matching_repos = filter_repos(repo_target=args.repo, dialect_target=args.dialect)
    if not matching_repos:
        sys.stderr.write(
            f"Error: No repositories matched filter (repo='{args.repo}', dialect='{args.dialect}')\n"
        )
        return 2

    log.info(f"Selected {len(matching_repos)} repository/repositories for validation.")

    if args.dry_run:
        log.info("[DRY-RUN] Planned validation steps:")
        dry_results: list[RepoResult] = []
        for entry in matching_repos:
            log.info(
                f"  - Repo: {entry.name} ({entry.slug}) | Dialect: {entry.dialect} | "
                f"Commit: {entry.commit[:10]} | Path: {entry.area_path} | Limit: {args.limit or 'all'}"
            )
            dry_results.append(
                RepoResult(
                    name=entry.name,
                    slug=entry.slug,
                    dialect=entry.dialect,
                    commit=entry.commit,
                    area_path=entry.area_path,
                    status="dry-run",
                )
            )

        if args.summary_json:
            summary = build_summary_dict(dry_results, dry_run=True)
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log.info(f"[DRY-RUN] Summary JSON written to {args.summary_json}")

        print_summary_table(dry_results)
        return 0

    results: list[RepoResult] = []
    for entry in matching_repos:
        log.info(f"Validating {entry.name} ({entry.dialect})...")
        result = validate_repository(
            entry=entry,
            cache_dir=args.cache_dir,
            limit=args.limit,
            verbose=args.verbose,
        )
        results.append(result)

    print_summary_table(results)

    if args.summary_json:
        summary = build_summary_dict(results, dry_run=False)
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info(f"Summary JSON written to {args.summary_json}")

    total_scanned = sum(r.files_scanned for r in results)
    total_failed = sum(r.files_failed for r in results)
    any_error = any(r.status == "error" for r in results)

    if any_error and total_scanned == 0:
        return 2
    if total_failed > 0:
        return 1
    if any_error:
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
