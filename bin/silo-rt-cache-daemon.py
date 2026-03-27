#!/usr/bin/env python3
"""
silo-rt-cache-daemon — Lease-aware cache daemon for rTorrent.

Polls rTorrent via XMLRPC on a configurable interval, writes display-ready
JSON rows to ~/.cache/silo-rt/torrents.json.  The silo dashboard reads from
this cache file instead of calling XMLRPC on every repaint.

URL resolution order:
  1. --xmlrpc-url CLI arg
  2. RTORRENT_XMLRPC_URL environment variable
  3. rtorrent.xmlrpc_url in config/silo.yml

Usage:
  silo-rt-cache-daemon [--xmlrpc-url URL] [--once] [--status]

Run as a background daemon — the dashboard calls ensure_daemon() automatically
when the rTorrent client is active and no daemon is running.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure bin/ is importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent))

import silo_client_rt as _rt
import silo_cache_common as _cc

_CACHE_BASE = Path.home() / ".cache" / "silo-rt"

# Config paths searched in order when resolving the XMLRPC URL from silo.yml
_SILO_YML_CANDIDATES = [
    Path(__file__).resolve().parents[1] / "config" / "silo.yml",
    Path.home() / ".config" / "silo" / "silo.yml",
]


def _resolve_url(arg_url: str) -> str:
    if arg_url:
        return arg_url
    env = os.environ.get("RTORRENT_XMLRPC_URL", "").strip()
    if env:
        return env
    try:
        import yaml  # type: ignore
        for p in _SILO_YML_CANDIDATES:
            if p.exists():
                data = yaml.safe_load(p.read_text()) or {}
                url = (data.get("downloaders") or {}).get("rtorrent", {}).get("xmlrpc_url", "")
                if url:
                    return url
    except Exception:
        pass
    return "http://localhost:18000/RPC2"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="silo-rt-cache-daemon",
        description="rTorrent cache daemon for silo",
    )
    p.add_argument("--xmlrpc-url",       default="",
                   help="rTorrent XMLRPC HTTP endpoint (overrides env/silo.yml)")
    p.add_argument("--cache-file",       default=str(_CACHE_BASE / "torrents.json"))
    p.add_argument("--meta-file",        default=str(_CACHE_BASE / "torrents.meta.json"))
    p.add_argument("--lease-dir",        default=str(_CACHE_BASE / "leases"))
    p.add_argument("--pid-file",         default=str(_CACHE_BASE / "daemon.pid"))
    p.add_argument("--lock-file",        default=str(_CACHE_BASE / "daemon.lock"))
    p.add_argument("--default-interval", type=float, default=5.0,
                   help="Normal poll interval in seconds (default: 5)")
    p.add_argument("--min-interval",     type=float, default=3.0)
    p.add_argument("--max-interval",     type=float, default=30.0)
    p.add_argument("--idle-grace",       type=float, default=120.0,
                   help="Exit after this many idle seconds with no active leases")
    p.add_argument("--sleep-step",       type=float, default=0.5)
    p.add_argument("--once",             action="store_true",
                   help="Fetch once, write cache, then exit (useful for scripts)")
    p.add_argument("--status",           action="store_true",
                   help="Print cache/daemon status as JSON and exit")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    cache_file = Path(args.cache_file)
    meta_file  = Path(args.meta_file)
    lease_dir  = Path(args.lease_dir)
    pid_file   = Path(args.pid_file)
    lock_file  = Path(args.lock_file)

    if args.status:
        print(json.dumps(
            _cc.status_payload(
                cache_file=cache_file,
                meta_file=meta_file,
                lease_dir=lease_dir,
                pid_file=pid_file,
            ),
            indent=2,
        ))
        return 0

    url = _resolve_url(args.xmlrpc_url)

    def fetch() -> str:
        rows = _rt.fetch(url)
        # Treat an empty list as a connection failure so the daemon backs off
        # rather than overwriting a healthy cache with nothing.
        if not rows:
            raise RuntimeError(f"rTorrent returned empty result from {url}")
        return json.dumps(rows)

    return _cc.run_daemon(
        fetch_fn=fetch,
        cache_file=cache_file,
        meta_file=meta_file,
        lease_dir=lease_dir,
        pid_file=pid_file,
        lock_file=lock_file,
        default_interval=args.default_interval,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
        idle_grace=args.idle_grace,
        sleep_step=args.sleep_step,
        run_once=args.once,
        extra_meta={"xmlrpc_url": url},
    )


if __name__ == "__main__":
    raise SystemExit(main())
