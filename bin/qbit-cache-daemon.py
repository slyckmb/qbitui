#!/usr/bin/env python3
"""DEPRECATED shim — use silo-cache-daemon.py instead. Removal: 2026-07-01."""
import sys
print("WARNING: qbit-cache-daemon.py is deprecated — use silo-cache-daemon.py instead.", file=sys.stderr)
from silo_hashall_shared import exec_hashall_script
if __name__ == "__main__":
    exec_hashall_script("qb-cache-daemon.py", use_bypass=True)
