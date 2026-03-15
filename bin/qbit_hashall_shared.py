#!/usr/bin/env python3
"""Helpers for consuming hashall's shared qB tooling from qbitui."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_HASHALL_ROOT = Path("/home/michael/dev/work/hashall")
DEFAULT_HASHALL_CACHE_BASE = Path.home() / ".cache" / "hashall-qb"


def resolve_hashall_root() -> Path:
    candidates: list[Path] = []

    env_root = os.environ.get("HASHALL_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            DEFAULT_HASHALL_ROOT,
            Path.home() / "dev" / "work" / "hashall",
        ]
    )

    for root in candidates:
        if (root / "bin" / "qb-cache-agent.py").exists() and (root / "bin" / "qb-cache-daemon.py").exists():
            return root

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Unable to locate hashall qB tooling. "
        f"Set HASHALL_ROOT or ensure hashall exists at one of: {searched}"
    )


def resolve_hashall_script(script_name: str) -> Path:
    script_path = resolve_hashall_root() / "bin" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing hashall script: {script_path}")
    return script_path


def exec_hashall_script(script_name: str):
    hashall_root = resolve_hashall_root()
    script_path = hashall_root / "bin" / script_name
    env = os.environ.copy()
    hashall_src = str(hashall_root / "src")
    current_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = hashall_src if not current_pythonpath else f"{hashall_src}{os.pathsep}{current_pythonpath}"
    os.execve(sys.executable, [sys.executable, str(script_path), *sys.argv[1:]], env)
