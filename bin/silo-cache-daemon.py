#!/usr/bin/env python3
"""silo-cache-daemon — qBittorrent shared cache daemon (silo-native implementation)."""

__version__ = "1.0.0"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qb_cache_lib import daemon_main

sys.exit(daemon_main())
