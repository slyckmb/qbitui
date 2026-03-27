#!/usr/bin/env python3
"""DEPRECATED shim — use 'silo' or 'silo qbit' instead. Removal: 2026-07-01."""
import os, sys
from pathlib import Path
print("WARNING: qbit-dashboard.py is deprecated — use 'silo' or 'silo qbit' instead.", file=sys.stderr)
target = Path(__file__).resolve().parent / "silo-dashboard.py"
os.execve(sys.executable, [sys.executable, str(target), *sys.argv[1:]], os.environ)
