#!/usr/bin/env python3
"""
silo_client_rt — rTorrent data/action layer for the silo dashboard.

Pure fetch + actions module.  No TUI code.  Uses stdlib xmlrpc.client only.
Connection: HTTP XMLRPC endpoint, e.g. http://localhost:8080/RPC2
  (configure nginx/lighttpd to proxy the rTorrent SCGI socket via HTTP)

Returns display dicts whose keys exactly match what build_list_block() /
build_narrow_list_block() in silo-dashboard.py expect (same schema as
build_rows() output).
"""

import os
import xmlrpc.client
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    _LOCAL_TZ = ZoneInfo("America/New_York")
except Exception:
    _LOCAL_TZ = timezone.utc

# ── Module-level identity ────────────────────────────────────────────────────

NAME    = "rTorrent"
KEY     = "rtorrent"   # key used in active_client comparisons
VERSION = "1.2.0"

# ── rTorrent multicall fields ────────────────────────────────────────────────
#
# API: d.multicall.filtered(view, filter_cmd, filter_value_CONSUMED, field1, ...)
#
# IMPORTANT: arg3 (filter_value) is silently consumed and never returned as data.
# To work around this, _MULTICALL_FIELDS starts with "d.hash=" twice:
#   call args: ('', '', 'd.hash=', 'd.hash=', 'd.name=', ...)
#                       ^^^^^^^^ consumed    ^^^^^^^^ real data index 0
#
# The result list for each torrent therefore starts at the second d.hash= (index 0).
# d.tracker_domain= is NOT available on this rTorrent build — tracker field is "-".

_MULTICALL_FIELDS = [
    "d.hash=",           # [0]  info hash (uppercase from rTorrent)
    "d.name=",           # [1]  display name
    "d.size_bytes=",     # [2]  total size in bytes
    "d.completed_bytes=",# [3]  downloaded bytes
    "d.up.rate=",        # [4]  current upload rate bytes/sec
    "d.down.rate=",      # [5]  current download rate bytes/sec
    "d.ratio=",          # [6]  ratio × 1000 (integer)
    "d.state=",          # [7]  0=stopped, 1=started
    "d.is_active=",      # [8]  0=inactive, 1=active (transfers running)
    "d.message=",        # [9]  tracker/error message
    "d.custom1=",        # [10] ruTorrent label → used as category
    "d.peers_accounted=",# [11] total peer count
    "d.creation_date=",  # [12] unix timestamp torrent was added
    "d.directory=",      # [13] parent save directory (for single-file: dir containing file)
    "d.hashing=",        # [14] 0=not hashing, 1=hash-checking (d.check_hash in progress)
    "d.base_path=",      # [15] full path to content: file path (single) or dir path (multi)
]

# Indices into the data result list (0-based, post-consumption of arg3)
_F_HASH      = 0
_F_NAME      = 1
_F_SIZE      = 2
_F_COMPLETED = 3
_F_UPRATE    = 4
_F_DLRATE    = 5
_F_RATIO     = 6
_F_STATE     = 7
_F_ACTIVE    = 8
_F_MESSAGE   = 9
_F_CUSTOM1   = 10
_F_PEERS     = 11
_F_CREATED   = 12
_F_DIRECTORY = 13
_F_HASHING   = 14
_F_BASE_PATH = 15

# ── Formatting helpers (mirrors dashboard equivalents) ───────────────────────

def _size_str(value) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except Exception:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def _speed_str(value) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except Exception:
        return "-"
    if value <= 0:
        return "0"
    return f"{_size_str(value)}/s"


def _eta_str(seconds: int) -> str:
    if seconds <= 0 or seconds >= 8640000:
        return "-"
    mins, sec = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m"
    return f"{mins}m"


def _added_str(ts) -> str:
    if not ts or int(ts) <= 0:
        return "-"
    try:
        return datetime.fromtimestamp(int(ts), _LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "-"


def _added_short_str(ts) -> str:
    if not ts or int(ts) <= 0:
        return "-"
    try:
        return datetime.fromtimestamp(int(ts), _LOCAL_TZ).strftime("%m-%d %H:%M")
    except Exception:
        return "-"


# ── State normalization ──────────────────────────────────────────────────────

def _normalize_state(state: int, is_active: int, message: str,
                     size_bytes: int, completed_bytes: int,
                     dl_rate: int, up_rate: int = 0, hashing: int = 0) -> str:
    """
    Map rTorrent state integers to a qBit-compatible API state string.

    The returned string must be a key in STATUS_MAPPING (silo-dashboard.py)
    so ColorScheme.status_color() maps it to the right color automatically.

    rTorrent state logic (evaluated top-to-bottom):
      hashing=1                              → "checkingDL" / "checkingUP"
      message != ""                          → "error"
      state=0, completed                     → "stoppedUP"  (seeded then stopped)
      state=0                                → "stoppedDL"  (paused group)
      state=1, is_active=0, completed        → "pausedUP"
      state=1, is_active=0                   → "pausedDL"
      state=1, is_active=1, size=0, dl>0     → "metaDL"     (magnet, no metadata yet)
      state=1, is_active=1, dl_rate > 0      → "downloading"
      state=1, is_active=1, completed, up>0  → "uploading"
      state=1, is_active=1, completed        → "stalledUP"  (seeding, no peers)
      else                                   → "stalledDL"
    """
    completed = size_bytes > 0 and completed_bytes >= size_bytes
    if hashing:
        return "checkingUP" if completed else "checkingDL"
    if message:
        return "error"
    if state == 0:
        return "stoppedUP" if completed else "stoppedDL"
    if is_active == 0:
        return "pausedUP" if completed else "pausedDL"
    # state=1, is_active=1 — active
    if size_bytes == 0 and dl_rate > 0:
        return "metaDL"   # magnet still downloading metadata
    if dl_rate > 0:
        return "downloading"
    if completed:
        return "uploading" if up_rate > 0 else "stalledUP"
    return "stalledDL"


# Map full state strings to short 2-char codes (mirrors STATE_CODE in dashboard)
_STATE_CODE = {
    "downloading":  "D",
    "uploading":    "U",
    "pausedDL":     "PD",
    "pausedUP":     "PU",
    "stoppedDL":    "PD",  # renders same as pausedDL
    "stoppedUP":    "PU",
    "stalledDL":    "SD",
    "stalledUP":    "SU",
    "checkingDL":   "CD",
    "checkingUP":   "CU",
    "metaDL":       "MD",
    "error":        "E",
    "checking":     "CR",
}


# ── Connection ───────────────────────────────────────────────────────────────

def connect(url: str) -> xmlrpc.client.ServerProxy:
    """Return an xmlrpc ServerProxy for the given HTTP XMLRPC URL."""
    return xmlrpc.client.ServerProxy(url, allow_none=True)


# ── Fetch ────────────────────────────────────────────────────────────────────

def fetch(url: str) -> list[dict]:
    """
    Fetch all torrents from rTorrent via d.multicall2 and return display dicts
    whose keys exactly match the schema expected by build_list_block() in
    silo-dashboard.py (same as build_rows() output).

    Returns [] on any connection or XMLRPC error — caller shows last cached list.
    """
    try:
        proxy = connect(url)
        # d.multicall.filtered signature: (view, filter_cmd, filter_value_CONSUMED, field...)
        # arg3 is silently consumed — pass "d.hash=" twice so real data starts at index 0.
        # view="" = all downloads; filter_cmd="" = no filtering.
        results = proxy.d.multicall.filtered("", "", "d.hash=", *_MULTICALL_FIELDS)
    except Exception:
        return []

    rows = []
    for item in results:
        try:
            hash_val     = str(item[_F_HASH]).lower()  # normalise to lowercase
            name         = str(item[_F_NAME])
            size_bytes   = int(item[_F_SIZE])   or 0
            completed_b  = int(item[_F_COMPLETED]) or 0
            up_rate      = int(item[_F_UPRATE])  or 0
            dl_rate      = int(item[_F_DLRATE])  or 0
            ratio_raw    = int(item[_F_RATIO])   or 0   # stored as ratio * 1000
            state        = int(item[_F_STATE])   or 0
            is_active    = int(item[_F_ACTIVE])  or 0
            message      = str(item[_F_MESSAGE] or "")
            label        = str(item[_F_CUSTOM1] or "")  # ruTorrent label
            peers        = int(item[_F_PEERS])   or 0
            created      = item[_F_CREATED]
            directory    = str(item[_F_DIRECTORY] or "")
            hashing      = int(item[_F_HASHING]) if len(item) > _F_HASHING else 0
            base_path    = str(item[_F_BASE_PATH] or "") if len(item) > _F_BASE_PATH else ""

            # Derived values
            ratio_float  = ratio_raw / 1000.0
            api_state    = _normalize_state(
                state, is_active, message, size_bytes, completed_b,
                dl_rate, up_rate, hashing
            )
            progress_f   = (completed_b / size_bytes) if size_bytes > 0 else 0.0
            progress_pct = int(progress_f * 100)

            eta_secs = 0
            if dl_rate > 0 and size_bytes > completed_b:
                eta_secs = (size_bytes - completed_b) // dl_rate

            category = label or "-"
            tracker  = "-"  # d.tracker_domain= not available on this rTorrent build

            # raw sub-dict: must have state/progress/dlspeed/upspeed so that
            # draw_header_full_compact() can compute aggregate stats from
            # rt_cached_torrents = [r["raw"] for r in rows]
            raw = {
                "hash":      hash_val,
                "name":      name,
                "state":     api_state,
                "progress":  progress_f,
                "dlspeed":   dl_rate,
                "upspeed":   up_rate,
                "size":      size_bytes,
                "ratio":     ratio_float,
                "eta":       eta_secs,      # sort_key reads raw.get("eta")
                "category":  category,
                "tags":      label or "-",
                "added_on":  int(created) if created else 0,
                "tracker":   tracker,
                "directory": directory,
                "base_path": base_path,   # d.base_path= = full path to file or dir
                "message":   message,
            }

            rows.append({
                "name":         name,
                "save_path":    directory.rstrip("/") or "-",
                "nohl":         " ",  # rTorrent has no ~nohl concept
                "state":        api_state,
                "st":           _STATE_CODE.get(api_state, "?"),
                "progress":     f"{progress_pct}%",
                "progress_pct": str(progress_pct),
                "size":         _size_str(size_bytes),
                "ratio":        f"{ratio_float:.2f}",
                "dlspeed":      _speed_str(dl_rate),
                "upspeed":      _speed_str(up_rate),
                "uploaded_raw": 0,   # rTorrent session upload not in multicall
                "seeds":        0,   # no seeds/peers split in multicall; use peers
                "peers":        peers,
                "eta":          _eta_str(eta_secs),
                "added":        _added_str(created),
                "added_short":  _added_short_str(created),
                "tracker":      tracker,
                "category":     category,
                "tags":         label or "-",
                "hash":         hash_val,
                "raw":          raw,
            })
        except Exception:
            continue  # skip malformed entries; don't crash the whole fetch

    return rows


# ── Per-torrent detail fetchers ──────────────────────────────────────────────

def fetch_trackers(proxy: xmlrpc.client.ServerProxy, hash_val: str) -> list:
    """Return tracker list as dicts with url/status/tier (matches render_trackers_lines)."""
    try:
        results = proxy.t.multicall(hash_val.upper(), "",
            "t.url=", "t.type=", "t.is_usable=", "t.scrape_success=") or []
        out = []
        for row in results:
            url, type_, is_usable, scrape_ok = row[0], row[1], int(row[2]), int(row[3])
            if is_usable and scrape_ok:
                status = "working"
            elif is_usable:
                status = "not working"
            else:
                status = "disabled"
            out.append({"url": url, "status": status, "tier": str(type_)})
        return out
    except Exception:
        return []


def fetch_files(proxy: xmlrpc.client.ServerProxy, hash_val: str) -> list:
    """Return file list as dicts with name/size/progress/priority (matches render_files_lines)."""
    try:
        results = proxy.f.multicall(hash_val.upper(), "",
            "f.path=", "f.size_bytes=", "f.completed_chunks=",
            "f.size_chunks=", "f.priority=") or []
        out = []
        for row in results:
            path, size_bytes, completed, total_chunks, priority = row
            progress = completed / total_chunks if total_chunks else 0.0
            out.append({
                "name":     path,
                "size":     int(size_bytes),
                "progress": progress,
                "priority": int(priority),
            })
        return out
    except Exception:
        return []


def fetch_peers(proxy: xmlrpc.client.ServerProxy, hash_val: str) -> dict:
    """Return peer dict in format expected by render_peers_lines."""
    try:
        results = proxy.p.multicall(hash_val.upper(), "",
            "p.address=", "p.port=", "p.client_version=",
            "p.down_rate=", "p.up_rate=", "p.completed_percent=",
            "p.is_incoming=") or []
        peers: dict = {}
        for row in results:
            addr, port, client, dl_rate, ul_rate, pct, is_incoming = row
            key = f"{addr}:{port}"
            peers[key] = {
                "dl_speed": int(dl_rate),
                "up_speed": int(ul_rate),
                "client":   str(client),
                "progress": float(pct) / 100.0,
                "flags":    "I" if is_incoming else "",
            }
        return {"peers": peers}
    except Exception:
        return {}


# ── Actions ──────────────────────────────────────────────────────────────────

def _action_pause_resume(proxy: xmlrpc.client.ServerProxy,
                          hash_val: str, item: dict) -> str:
    """Pause if active, resume if paused/stopped."""
    try:
        state   = item.get("raw", {}).get("state", "")
        paused  = state in ("pausedDL", "pausedUP", "stoppedDL", "stoppedUP")
        if paused:
            proxy.d.start(hash_val.upper())
            return "Resumed"
        else:
            proxy.d.stop(hash_val.upper())
            return "Paused"
    except Exception as e:
        return f"Error: {e}"


def _action_verify(proxy: xmlrpc.client.ServerProxy,
                   hash_val: str, item: dict) -> str:
    """Trigger hash check."""
    try:
        proxy.d.check_hash(hash_val.upper())
        return "Hash check started"
    except Exception as e:
        return f"Error: {e}"


def _action_delete(proxy: xmlrpc.client.ServerProxy,
                   hash_val: str, item: dict) -> str:
    """
    Remove torrent from rTorrent (data is NOT deleted).
    Uses d.erase — equivalent to 'Remove torrent' in ruTorrent.
    To also delete data, a separate filesystem call would be needed.
    """
    name = item.get("name", hash_val[:8])
    # Inline y/N prompt — raw mode is restored by caller before this runs
    try:
        answer = input(f"Remove '{name}' from rTorrent? Data kept. [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return "Cancelled"
    if answer != "y":
        return "Cancelled"
    try:
        proxy.d.erase(hash_val.upper())
        return "Removed"
    except Exception as e:
        return f"Error: {e}"


# ACTIONS dict: key → (display_label, fn(proxy, hash_val, item) -> str)
# Keys use the same letters as qBit actions where semantically identical.
ACTIONS: dict[str, tuple[str, object]] = {
    "P": ("Pause/Resume", _action_pause_resume),
    "V": ("Verify",       _action_verify),
    "D": ("Delete",       _action_delete),
}
