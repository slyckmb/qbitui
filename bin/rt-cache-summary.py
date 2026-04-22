#!/usr/bin/env python3
"""Summarize the shared silo rTorrent cache without touching rTorrent directly."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_DAEMON = Path("/home/michael/dev/tools/silo/bin/silo-rt-cache-daemon.py")
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "silo-rt"
SCRIPT_NAME = "rt-cache-summary.py"
SCRIPT_VERSION = "1.0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE_DIR / "torrents.json"))
    parser.add_argument("--meta-file", default=str(DEFAULT_CACHE_DIR / "torrents.meta.json"))
    parser.add_argument("--daemon", default=str(DEFAULT_DAEMON))
    parser.add_argument("--python", dest="python_bin", default="python3")
    parser.add_argument("--status-timeout", type=float, default=5.0)
    parser.add_argument("--max-age", type=float, default=60.0)
    return parser.parse_args()


def read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_status(args: argparse.Namespace) -> dict[str, Any]:
    daemon = Path(args.daemon)
    if not daemon.is_file():
        return {}
    try:
        proc = subprocess.run(
            [args.python_bin, str(daemon), "--status"],
            capture_output=True,
            text=True,
            timeout=args.status_timeout,
            check=False,
        )
    except Exception as exc:
        return {"status_error": str(exc)}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "status_error": proc.stderr.strip() or f"daemon status exited {proc.returncode}"
        }
    try:
        parsed = json.loads(proc.stdout)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError as exc:
        return {"status_error": f"invalid daemon status json: {exc}"}
    return {"status_error": "unexpected daemon status payload"}


def to_int(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def classify_tracker_warning(message: str) -> str:
    msg = str(message or "").strip()
    if not msg:
        return ""
    if re.search(r"already have [38] peer|Sorry max peers|max peers reached", msg):
        return "peer_lim"
    if re.search(r"3 location|rate limit.*location", msg):
        return "multi_loc"
    if re.search(r"SSL|certificate|SSL peer", msg):
        return "ssl_err"
    if re.search(r"Timeout|Host not found|stream truncat|Connection reset|non-authorit", msg):
        return "conn_err"
    if re.search(r"has been deleted", msg):
        return "deleted"
    if re.search(r"passkey|InfoHash|not found in [Hh]istory|Torrent not found", msg):
        return "auth_err"
    return "other"


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    states: Counter[str] = Counter()
    tracker_warn_by_kind: Counter[str] = Counter()
    peers_total = 0
    dl_speed = 0
    ul_speed = 0
    tracker_warn_total = 0
    error_fatal = 0
    for row in rows:
        state = str(row.get("state") or "").strip() or "unknown"
        states[state] += 1
        peers_total += to_int(row.get("peers"))
        dl_speed += to_int(row.get("dlspeed"))
        ul_speed += to_int(row.get("upspeed"))
        message = str(row.get("message") or row.get("raw", {}).get("message") or "").strip()
        if not message:
            continue
        if state == "error":
            error_fatal += 1
            continue
        if state in {"stalledUP", "uploading", "stoppedUP"}:
            tracker_warn_total += 1
            tracker_warn_by_kind[classify_tracker_warning(message)] += 1
    top_states = [f"{state}={count}" for state, count in states.most_common(8)]
    return {
        "items": len(rows),
        "states": dict(states),
        "error_fatal": error_fatal,
        "tracker_warn_total": tracker_warn_total,
        "tracker_warn_by_kind": dict(tracker_warn_by_kind),
        "top_states": top_states,
        "peers_total": peers_total,
        "dl_speed": dl_speed,
        "ul_speed": ul_speed,
    }


def compute_age_s(now: float, status: dict[str, Any], meta: dict[str, Any], cache_file: Path) -> float | None:
    cache_age = status.get("cache_age_s")
    if cache_age is not None:
        try:
            return float(cache_age)
        except Exception:
            pass
    fetched_at = meta.get("fetched_at")
    if fetched_at is not None:
        try:
            return max(0.0, now - float(fetched_at))
        except Exception:
            pass
    if cache_file.exists():
        try:
            return max(0.0, now - cache_file.stat().st_mtime)
        except OSError:
            return None
    return None


def main() -> int:
    args = parse_args()
    now = time.time()
    cache_file = Path(args.cache_file).expanduser()
    meta_file = Path(args.meta_file).expanduser()

    status_payload = load_status(args)
    meta = status_payload.get("meta")
    if not isinstance(meta, dict):
        meta = read_json_file(meta_file)
        if not isinstance(meta, dict):
            meta = {}

    rows_payload = read_json_file(cache_file)
    rows: list[dict[str, Any]]
    if isinstance(rows_payload, list):
        rows = [row for row in rows_payload if isinstance(row, dict)]
    else:
        rows = []

    cache_age_s = compute_age_s(now, status_payload, meta, cache_file)
    cache_source = str(meta.get("source") or "")
    daemon_running = bool(status_payload.get("daemon_running"))
    active_leases = to_int(status_payload.get("active_lease_count") or meta.get("active_leases"))
    last_error = (
        str(meta.get("last_error") or status_payload.get("last_error") or status_payload.get("status_error") or "")
    )

    if rows:
        freshness = "fresh"
        if cache_age_s is not None and cache_age_s > args.max_age:
            freshness = "stale"
        if cache_source == "daemon_error":
            freshness = "stale_error" if rows else "error"
    else:
        freshness = "missing"

    summary = summarize_rows(rows)
    payload = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "ok": bool(rows),
        "freshness": freshness,
        "cache_source": cache_source,
        "cache_age_s": cache_age_s,
        "max_age_s": float(args.max_age),
        "daemon_running": daemon_running,
        "active_leases": active_leases,
        "status_error": status_payload.get("status_error"),
        "last_error": last_error,
        "cache_file": str(cache_file),
        "meta_file": str(meta_file),
        "xmlrpc_url": meta.get("xmlrpc_url") or status_payload.get("xmlrpc_url"),
        "items": summary["items"],
        "states": summary["states"],
        "error_fatal": summary["error_fatal"],
        "tracker_warn_total": summary["tracker_warn_total"],
        "tracker_warn_by_kind": summary["tracker_warn_by_kind"],
        "top_states": summary["top_states"],
        "peers_total": summary["peers_total"],
        "dl_speed": summary["dl_speed"],
        "ul_speed": summary["ul_speed"],
        "sockets": None,
    }
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
