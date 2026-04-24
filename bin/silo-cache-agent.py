#!/usr/bin/env python3
"""silo-cache-agent — qBittorrent shared cache agent (silo-native implementation)."""

__version__ = "1.0.0"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qb_cache_lib import agent_main

sys.exit(agent_main())
