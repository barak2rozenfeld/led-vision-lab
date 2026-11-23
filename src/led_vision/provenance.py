"""Provenance info attached to generated reports (stats, benchmarks)."""
from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Paths whose uncommitted changes can change a report's numbers.
RESULT_AFFECTING_PATHS: tuple[str, ...] = ("src", "configs", "scripts")


def _git(repo_dir: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip()


def git_commit(repo_dir: Path) -> str | None:
    """Short commit hash, or None when git is unavailable or this is not a repository."""
    return _git(repo_dir, "rev-parse", "--short", "HEAD") or None


def git_dirty(repo_dir: Path) -> bool | None:
    """True when code or configs differ from the commit, so the hash alone can't reproduce the report."""
    status = _git(repo_dir, "status", "--porcelain", "--", *RESULT_AFFECTING_PATHS)
    return None if status is None else bool(status)


def provenance(repo_dir: Path) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(repo_dir),
        "git_dirty": git_dirty(repo_dir),
        "machine": f"{platform.system()} {platform.machine()} · {platform.processor() or 'cpu'}",
        "python": platform.python_version(),
    }
