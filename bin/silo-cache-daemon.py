#!/usr/bin/env python3
"""
qbit-cache-daemon.py — DEPRECATED SHIM

This script has moved to qbitui (canonical location):
  /home/michael/dev/tools/qbitui/bin/qbit-cache-daemon.py

This shim execs the canonical script when available so that existing hashall
callers continue to work without any command-line changes.

Migration: update your scripts to call the canonical path directly.
"""
import os
import sys

CANONICAL = "/home/michael/dev/tools/qbitui/bin/qbit-cache-daemon.py"

if os.path.exists(CANONICAL):
    os.execv(sys.executable, [sys.executable, CANONICAL] + sys.argv[1:])

print(
    f"qbit-cache-daemon shim: canonical script not found at {CANONICAL!r}; "
    "please install qbitui or update CANONICAL path in this shim.",
    file=sys.stderr,
)
sys.exit(1)
