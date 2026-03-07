#!/usr/bin/env python3
"""
qb-cache-agent.py — DEPRECATED SHIM

This script has moved to qbitui (canonical location):
  /home/michael/dev/tools/qbitui/bin/qb-cache-agent.py

This shim execs the canonical script when available so that existing hashall
callers (qb-start-seeding-gradual.sh, qb-checking-watch.sh, etc.)
continue to work without any command-line changes.

Migration: update your scripts to call the canonical path directly.
"""
import os
import sys

CANONICAL = "/home/michael/dev/tools/qbitui/bin/qb-cache-agent.py"

if os.path.exists(CANONICAL):
    os.execv(sys.executable, [sys.executable, CANONICAL] + sys.argv[1:])

# Canonical not found — warn and fall back to local copy if present.
print(
    f"qb-cache-agent shim: canonical script not found at {CANONICAL!r}; "
    "please install qbitui or update CANONICAL path in this shim.",
    file=sys.stderr,
)
sys.exit(1)
