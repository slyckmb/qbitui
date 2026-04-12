#!/usr/bin/env python3
"""Lease-aware helper for the local hashall qB shared cache."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hashall.qb_cache import agent_main
from hashall.script_metadata import register as register_script_metadata

SCRIPT_NAME = Path(__file__).name
SEMVER = "0.1.0"
LAST_UPDATED = "2026-04-09T07:05:00-04:00"
register_script_metadata(SCRIPT_NAME, SEMVER, LAST_UPDATED, argv=" ".join(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(agent_main())
