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
import time
import urllib.parse
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
VERSION = "1.3.1"

# ── Module-level tracker cache ───────────────────────────────────────────────
# Populated by _fetch_tracker_batch() inside fetch(); persists across daemon
# fetch cycles (the daemon imports this module once and keeps it in memory).
# Refresh happens at most once per _TRACKER_CACHE_TTL seconds.

_tracker_cache: dict      = {}   # hash_lower -> list[dict]  (url/status/tier)
_tracker_cache_time: float = 0.0
_TRACKER_CACHE_TTL: float  = 300.0   # seconds between full tracker refreshes
_TRACKER_BATCH_SIZE: int   = 500     # t.multicall calls per system.multicall chunk

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
    "d.directory=",      # [13] save directory (multi-file: already includes torrent name)
    "d.hashing=",        # [14] 0=not hashing, 1=hash-checking (d.check_hash in progress)
    # NOTE: d.base_path= and d.tracker_domain= are NOT available on this rTorrent build.
    # base_path is computed in Python from directory + name (see _compute_base_path).
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

# ── Tracker batch fetch ───────────────────────────────────────────────────────

def _url_domain(url: str) -> str:
    """Extract hostname from a tracker URL ('http://tracker.example.com:6969/ann' → 'tracker.example.com')."""
    try:
        return urllib.parse.urlparse(url).hostname or "-"
    except Exception:
        return "-"


def _fetch_tracker_batch(
    proxy: xmlrpc.client.ServerProxy,
    hashes: list,
) -> dict:
    """Fetch tracker data for all hashes via system.multicall, in chunks.

    Returns dict: hash_lower -> list[dict{url, status, tier}].
    Each chunk bundles _TRACKER_BATCH_SIZE t.multicall calls into one HTTP
    request, keeping RT XMLRPC load low even for large swarms.

    Fields used: t.url= and t.is_usable= — confirmed available on this build.
    t.scrape_success= is NOT available and must not be requested.
    Status mapping: is_usable=1 → "working", is_usable=0 → "disabled".
    """
    result: dict = {}
    for i in range(0, len(hashes), _TRACKER_BATCH_SIZE):
        chunk = hashes[i : i + _TRACKER_BATCH_SIZE]
        calls = [
            {
                "methodName": "t.multicall",
                "params": [h.upper(), "", "t.url=", "t.is_usable="],
            }
            for h in chunk
        ]
        try:
            responses = proxy.system.multicall(calls)
        except Exception:
            # Partial failure: skip this chunk, leave hashes unmapped
            continue
        for h, resp in zip(chunk, responses):
            # system.multicall wraps each result in a 1-element list on success,
            # or returns a fault dict on per-call error.
            if isinstance(resp, dict) and "faultCode" in resp:
                result[h] = []
                continue
            rows = resp[0] if (resp and isinstance(resp, list)) else []
            trackers = []
            for row in rows:
                try:
                    url    = str(row[0])
                    usable = int(row[1])
                    status = "working" if usable else "disabled"
                    trackers.append({"url": url, "status": status, "tier": "0"})
                except Exception:
                    continue
            result[h] = trackers
    return result


# ── Path helpers ─────────────────────────────────────────────────────────────

def _compute_base_path(directory: str, name: str) -> str:
    """Derive the full content path without relying on d.base_path=.

    rTorrent stores:
      - Multi-file torrent: d.directory= = /base/TorrentName  (name already appended)
      - Single-file torrent: d.directory= = /base             (name is the filename)

    Strategy: try directory/name as a file first; if it exists that is the
    single-file case.  Otherwise fall back to directory itself (multi-file).
    """
    if not directory:
        return ""
    candidate = Path(directory) / name if name else None
    if candidate and candidate.is_file():
        return str(candidate)
    return directory


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
    Fetch all torrents from rTorrent via d.multicall.filtered and return display
    dicts whose keys exactly match the schema expected by build_list_block() in
    silo-dashboard.py (same as build_rows() output).

    Tracker data is refreshed from RT via system.multicall at most once every
    _TRACKER_CACHE_TTL seconds (default 300 s), then merged into each row's
    'raw' sub-dict without additional per-call XMLRPC traffic.

    Returns [] on any connection or XMLRPC error — caller shows last cached list.
    """
    global _tracker_cache, _tracker_cache_time
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
            base_path    = _compute_base_path(directory, name)

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
            complete = size_bytes > 0 and completed_b >= size_bytes

            # raw sub-dict: must have state/progress/dlspeed/upspeed so that
            # draw_header_full_compact() can compute aggregate stats from
            # rt_cached_torrents = [r["raw"] for r in rows].
            # tracker/trackers* fields are filled in after the loop via the
            # module-level TTL tracker cache (_tracker_cache).
            raw = {
                "hash":              hash_val,
                "name":              name,
                "state":             api_state,
                "progress":          progress_f,
                "dlspeed":           dl_rate,
                "upspeed":           up_rate,
                "size":              size_bytes,
                "ratio":             ratio_float,
                "eta":               eta_secs,
                "category":          category,
                "tags":              label or "-",
                "added_on":          int(created) if created else 0,
                "tracker":           "-",   # filled after tracker cache refresh
                "trackers":          [],    # filled after tracker cache refresh
                "trackers_http":     [],    # filled after tracker cache refresh
                "trackers_count":    0,     # filled after tracker cache refresh
                "real_trackers_count": 0,   # filled after tracker cache refresh
                "directory":         directory,
                "base_path":         base_path,
                "message":           message,
                "complete":          complete,
                "hashing":           hashing,
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
                "tracker":      "-",  # filled after tracker cache refresh
                "category":     category,
                "tags":         label or "-",
                "hash":         hash_val,
                "raw":          raw,
            })
        except Exception:
            continue  # skip malformed entries; don't crash the whole fetch

    # ── Tracker enrichment ────────────────────────────────────────────────────
    # Refresh module-level tracker cache when TTL has expired, then merge
    # tracker data into each row's raw dict without per-torrent XMLRPC calls.
    if rows:
        now = time.time()
        if now - _tracker_cache_time >= _TRACKER_CACHE_TTL:
            try:
                hashes = [r["hash"] for r in rows]
                _tracker_cache = _fetch_tracker_batch(proxy, hashes)
                _tracker_cache_time = now
            except Exception:
                pass  # keep stale cache; enrich with whatever we have

        for row in rows:
            h        = row["hash"]
            trackers = _tracker_cache.get(h, [])
            working  = [t for t in trackers if t["status"] == "working"]
            # Primary tracker: first working, else first any, else empty
            primary_url = (
                working[0]["url"] if working
                else trackers[0]["url"] if trackers
                else ""
            )
            domain = _url_domain(primary_url) if primary_url else "-"
            http_urls = [t["url"] for t in trackers if t["url"].startswith("http")]

            raw = row["raw"]
            raw["tracker"]             = domain
            raw["trackers"]            = trackers
            raw["trackers_http"]       = http_urls
            raw["trackers_count"]      = len(trackers)
            raw["real_trackers_count"] = len(working)
            row["tracker"]             = domain

    return rows


# ── Per-torrent detail fetchers ──────────────────────────────────────────────

def fetch_trackers(proxy: xmlrpc.client.ServerProxy, hash_val: str) -> list:
    """Return tracker list as dicts with url/status/tier (matches render_trackers_lines).

    Uses t.url=, t.is_usable=, t.failed_counter= — all confirmed available on
    this rTorrent build.  t.scrape_success= is NOT available and must not be
    requested.

    Status mapping:
      is_usable=0                       → "disabled"
      is_usable=1, failed_counter=0     → "working"
      is_usable=1, failed_counter>0     → "not working"
    """
    try:
        results = proxy.t.multicall(hash_val.upper(), "",
            "t.url=", "t.is_usable=", "t.failed_counter=") or []
        out = []
        for row in results:
            url          = str(row[0])
            is_usable    = int(row[1])
            failed_count = int(row[2])
            if not is_usable:
                status = "disabled"
            elif failed_count == 0:
                status = "working"
            else:
                status = "not working"
            out.append({"url": url, "status": status, "tier": "0"})
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
