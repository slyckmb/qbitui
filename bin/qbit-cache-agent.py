#!/usr/bin/env python3
"""DEPRECATED shim — use silo-cache-agent.py instead. Removal: 2026-07-01."""
import sys
print("WARNING: qbit-cache-agent.py is deprecated — use silo-cache-agent.py instead.", file=sys.stderr)
from silo_hashall_shared import exec_hashall_script
if __name__ == "__main__":
    exec_hashall_script("qb-cache-agent.py", use_bypass=True)
