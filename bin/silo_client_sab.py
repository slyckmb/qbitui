#!/usr/bin/env python3
"""
silo_client_sab.py — SABnzbd data/action layer for the silo suite.

No TUI code.  Imported by:
  - silo-dashboard.py  (unified multi-client dashboard)
  - silo-sabnzbd.py    (standalone SABnzbd TUI)
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

NAME = "SABnzbd"
KEY  = "sabnzbd"

last_error: str = ""   # cleared on success; callers may display it

# ---------------------------------------------------------------------------
# State mapping
# ---------------------------------------------------------------------------

# qBit-compatible state strings → 2-char display codes used by the dashboard
_STATE_CODE: dict[str, str] = {
    "downloading": "DL",
    "pausedDL":    "PD",
    "stalledDL":   "SD",
    "checkingDL":  "CD",
    "error":       "ER",
    "stoppedUP":   "SU",   # completed history item
    "stoppedDL":   "SP",   # stopped in queue without completing
}


def _normalize_state(status: str, source: str) -> str:
    """Map a SABnzbd status string to a qBit-compatible state string."""
    s = (status or "").lower().strip()

    if source == "H":
        if "fail" in s or "error" in s:
            return "error"
        return "stoppedUP"   # Completed, Downloaded, etc.

    # Queue items
    if "fail" in s or "error" in s:
        return "error"
    if "pause" in s:
        return "pausedDL"
    if "check" in s or "verif" in s:
        return "checkingDL"
    if s in ("queued", "idle", "propagating"):
        return "stalledDL"
    if "download" in s or "fetch" in s or "grab" in s or "extract" in s \
            or "repair" in s or "unpack" in s or s == "active":
        return "downloading"
    return "stalledDL"


# ---------------------------------------------------------------------------
# Connection object
# ---------------------------------------------------------------------------

class SabConn:
    """Lightweight connection descriptor.  No persistent socket."""

    def __init__(self, api_url: str, api_key: str) -> None:
        self.api_url = api_url.rstrip("/")
        if not self.api_url.endswith("/api"):
            self.api_url = self.api_url + "/api"
        self.api_key = api_key

    def __repr__(self) -> str:  # pragma: no cover
        return f"SabConn({self.api_url!r})"


def connect(api_url: str, api_key: str) -> SabConn:
    """Return a SabConn.  Does not open a socket."""
    return SabConn(api_url, api_key)


# ---------------------------------------------------------------------------
# HTTP request
# ---------------------------------------------------------------------------

def request(conn: SabConn, params: dict, timeout: int = 10) -> dict:
    """Make a SABnzbd API call and return the parsed JSON dict.

    On any error, sets ``last_error`` and returns ``{}``.
    Clears ``last_error`` on success.
    """
    global last_error
    url = (
        f"{conn.api_url}?"
        f"{urllib.parse.urlencode({**params, 'apikey': conn.api_key, 'output': 'json'})}"
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        is_reset = isinstance(reason, OSError) and getattr(reason, "errno", None) == 104
        if is_reset:
            last_error = "Connection reset by peer (errno 104)"
        elif isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            last_error = "Request timed out"
        else:
            last_error = f"Network error: {reason}"
        return {}
    except OSError as exc:
        last_error = f"OS error: {exc}"
        return {}
    except Exception as exc:
        last_error = f"Unexpected error: {exc}"
        return {}

    stripped = body.strip()
    if stripped.startswith("<!DOCTYPE") or stripped.startswith("<html"):
        last_error = "Got HTML instead of JSON (check API URL / key)"
        return {}
    try:
        data = json.loads(body)
        last_error = ""
        return data
    except json.JSONDecodeError:
        last_error = f"Invalid JSON response ({len(body)} bytes)"
        return {}


# ---------------------------------------------------------------------------
# Size / ETA / age helpers
# ---------------------------------------------------------------------------

_SIZE_UNITS = {
    "b":  1,
    "kb": 1_024,
    "mb": 1_048_576,
    "gb": 1_073_741_824,
    "tb": 1_099_511_627_776,
}


def _parse_size_bytes(value) -> int:
    """Parse a SABnzbd size value to bytes.

    Accepts int/float (treated as bytes) or strings like "1.5 GB", "500 MB".
    """
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value or "").strip().lower()
    if not s or s == "-":
        return 0
    try:
        parts = s.rsplit(None, 1)
        if len(parts) == 2:
            num = float(parts[0].replace(",", ""))
            unit = parts[1].rstrip("s")
            return int(num * _SIZE_UNITS.get(unit, 1))
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _parse_eta_seconds(timeleft: str) -> int:
    """Parse a SABnzbd timeleft string "H:MM:SS" → total seconds."""
    s = str(timeleft or "").strip()
    if not s or s in ("-", "0:00:00"):
        return 0
    try:
        parts = [int(x) for x in s.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        return int(parts[0])
    except (ValueError, TypeError):
        return 0


def _age_str(value) -> str:
    """Convert a Unix timestamp or ISO string to a human-readable age."""
    if not value:
        return "n/a"
    dt = None
    if isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, timezone.utc)
    else:
        try:
            dt = datetime.fromtimestamp(int(value), timezone.utc)
        except Exception:
            try:
                dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except Exception:
                dt = None
    if not dt:
        return "n/a"
    now = datetime.now(timezone.utc)
    delta = now - dt
    minutes = int(delta.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"


# ---------------------------------------------------------------------------
# fetch() — unified silo schema (for silo-dashboard.py)
# ---------------------------------------------------------------------------

def fetch(conn: SabConn, history_limit: int = 25) -> list[dict]:
    """Fetch queue + history and return rows in the unified silo display schema.

    Schema keys match what ``build_list_block()`` and ``draw_header_v2()``
    expect (same as ``silo_client_rt.fetch()``):
      name, save_path, nohl, state, st, progress, progress_pct, size, ratio,
      dlspeed, upspeed, seeds, peers, eta, added, added_short, tracker,
      category, tags, hash, source, raw

    ``raw`` sub-dict always contains integer ``dlspeed`` / ``upspeed`` keys so
    that ``draw_header_v2`` can sum aggregate bandwidth.

    Returns [] on total connection failure (caller checks ``last_error``).
    """
    queue_data  = request(conn, {"mode": "queue"})
    history_data = request(conn, {"mode": "history", "limit": history_limit})
    if not queue_data and not history_data:
        return []

    q = (queue_data.get("queue") or {}) if queue_data else {}
    # SABnzbd reports one aggregate download speed; assign it to the first
    # actively-downloading queue slot so the header total is correct.
    queue_speed_bps = int(float(q.get("kbpersec") or 0) * 1024)
    queue_slots = q.get("slots") or []

    rows: list[dict] = []

    for i, slot in enumerate(queue_slots):
        status_raw = str(slot.get("status") or "Queued")
        state = _normalize_state(status_raw, "Q")
        st    = _STATE_CODE.get(state, "??")

        pct      = int(slot.get("percentage") or 0)
        mb_total = float(slot.get("mb") or 0)
        size_bytes = int(mb_total * 1_048_576) if mb_total else 0

        dlspeed = queue_speed_bps if (i == 0 and state == "downloading") else 0
        eta_secs = _parse_eta_seconds(slot.get("timeleft"))

        name     = slot.get("filename") or slot.get("name") or ""
        nzo_id   = slot.get("nzo_id") or slot.get("nzoid") or ""
        category = slot.get("cat") or slot.get("category") or ""
        added_ts = slot.get("time_added") or 0

        raw = dict(slot)
        raw["dlspeed"] = dlspeed
        raw["upspeed"] = 0

        rows.append({
            "name":         name,
            "save_path":    slot.get("storage") or "",
            "nohl":         " ",
            "state":        state,
            "st":           st,
            "progress":     f"{pct}%",
            "progress_pct": float(pct),
            "size":         size_bytes,
            "ratio":        0.0,
            "dlspeed":      dlspeed,
            "upspeed":      0,
            "seeds":        0,
            "peers":        0,
            "eta":          eta_secs,
            "added":        _age_str(added_ts),
            "added_short":  _age_str(added_ts),
            "tracker":      "usenet",
            "category":     category,
            "tags":         category,
            "hash":         nzo_id,
            "source":       "Q",
            "raw":          raw,
        })

    h = (history_data.get("history") or {}) if history_data else {}
    for slot in (h.get("slots") or []):
        status_raw = str(slot.get("status") or "Completed")
        state = _normalize_state(status_raw, "H")
        st    = _STATE_CODE.get(state, "??")

        size_bytes = int(slot.get("bytes") or 0)
        if not size_bytes:
            size_bytes = _parse_size_bytes(slot.get("size"))

        name         = slot.get("name") or slot.get("filename") or ""
        nzo_id       = slot.get("nzo_id") or slot.get("nzoid") or ""
        category     = slot.get("cat") or slot.get("category") or ""
        completed_ts = slot.get("completed") or 0

        raw = dict(slot)
        raw["dlspeed"] = 0
        raw["upspeed"] = 0

        rows.append({
            "name":         name,
            "save_path":    slot.get("storage") or slot.get("path") or "",
            "nohl":         " ",
            "state":        state,
            "st":           st,
            "progress":     "100%",
            "progress_pct": 100.0,
            "size":         size_bytes,
            "ratio":        0.0,
            "dlspeed":      0,
            "upspeed":      0,
            "seeds":        0,
            "peers":        0,
            "eta":          0,
            "added":        _age_str(completed_ts),
            "added_short":  _age_str(completed_ts),
            "tracker":      "usenet",
            "category":     category,
            "tags":         category,
            "hash":         nzo_id,
            "source":       "H",
            "raw":          raw,
        })

    return rows


# ---------------------------------------------------------------------------
# fetch_raw() / build_rows_native() — SABnzbd-native schema (for silo-sabnzbd.py)
# ---------------------------------------------------------------------------

def fetch_raw(conn: SabConn, history_limit: int = 25) -> tuple[dict, dict]:
    """Return (queue_data, history_data) raw API dicts for silo-sabnzbd.py."""
    queue_data   = request(conn, {"mode": "queue"}) or {}
    history_data = request(conn, {"mode": "history", "limit": history_limit}) or {}
    return queue_data, history_data


def build_rows_native(queue: dict, history: dict) -> list[dict]:
    """Build SABnzbd-native schema rows consumed by silo-sabnzbd.py.

    Row keys: source, status, name, progress, size, eta_age, category, id, raw
    """
    rows: list[dict] = []

    q = (queue.get("queue") or {}) if isinstance(queue, dict) else {}
    for slot in (q.get("slots") or []):
        name       = slot.get("filename") or slot.get("name") or ""
        status_raw = str(slot.get("status") or "")
        percent    = slot.get("percentage") or slot.get("percent") or ""
        if isinstance(percent, (int, float)):
            percent = f"{int(percent)}%"
        elif isinstance(percent, str) and percent and not percent.endswith("%"):
            percent = percent + "%"
        size     = slot.get("size") or slot.get("mb") or "-"
        eta      = slot.get("timeleft") or slot.get("time_left") or "-"
        category = slot.get("cat") or slot.get("category") or "-"
        nzo_id   = slot.get("nzo_id") or slot.get("nzoid") or ""
        rows.append({
            "source":   "Q",
            "status":   status_raw,
            "name":     name,
            "progress": percent or "-",
            "size":     size,
            "eta_age":  eta or "-",
            "category": category,
            "id":       nzo_id,
            "raw":      slot,
        })

    h = (history.get("history") or {}) if isinstance(history, dict) else {}
    for slot in (h.get("slots") or []):
        name       = slot.get("name") or slot.get("filename") or ""
        status_raw = str(slot.get("status") or "")
        size       = slot.get("size") or "-"
        category   = slot.get("cat") or slot.get("category") or "-"
        nzo_id     = slot.get("nzo_id") or slot.get("nzoid") or ""
        age        = _age_str(slot.get("completed"))
        rows.append({
            "source":   "H",
            "status":   status_raw,
            "name":     name,
            "progress": "100%",
            "size":     size,
            "eta_age":  age,
            "category": category,
            "id":       nzo_id,
            "raw":      slot,
        })

    return rows


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------

def summarize(queue: dict, history: dict) -> str:
    """Return a one-line summary of queue + history state."""
    q = (queue.get("queue") or {}) if isinstance(queue, dict) else {}
    status    = q.get("status") or q.get("state") or "unknown"
    speed     = q.get("speed") or q.get("kbpersec") or "-"
    size_left = q.get("sizeleft") or q.get("mbleft") or "-"
    time_left = q.get("timeleft") or q.get("time_left") or q.get("eta") or "-"
    slot_count = q.get("noofslots") or len(q.get("slots") or [])

    h = (history.get("history") or {}) if isinstance(history, dict) else {}
    slots = h.get("slots") or []
    hist_counts: dict[str, int] = {}
    for slot in slots:
        s = str(slot.get("status") or "unknown").lower()
        hist_counts[s] = hist_counts.get(s, 0) + 1

    summary = (
        f"queue:{slot_count} status:{status} speed:{speed} "
        f"left:{size_left} eta:{time_left}"
    )
    if slots:
        total_hist = sum(hist_counts.values())
        failed = hist_counts.get("failed", 0) + hist_counts.get("fail", 0)
        summary += f" history:{total_hist} failed:{failed}"
    return summary


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _action_pause_resume(conn: SabConn, item_id: str, item: dict) -> str:
    if item.get("source") == "H":
        return "Pause/resume is only for queue items"
    state = (item.get("state") or item.get("status") or "").lower()
    action = "resume" if "pause" in state else "pause"
    result = request(conn, {"mode": "queue", "name": action, "value": item_id})
    return "OK" if result.get("status") else (last_error or "Failed")


def _action_delete(conn: SabConn, item_id: str, item: dict) -> str:
    mode = "history" if item.get("source") == "H" else "queue"
    result = request(conn, {"mode": mode, "name": "delete", "value": item_id})
    return "OK" if result.get("status") else (last_error or "Failed")


def _action_retry(conn: SabConn, item_id: str, item: dict) -> str:
    if item.get("source") != "H":
        return "Retry is only for history items"
    result = request(conn, {"mode": "retry", "value": item_id})
    return "OK" if result.get("status") else (last_error or "Failed")


# Dashboard action table: key → (label, fn(conn, item_id, item) → str)
ACTIONS: dict[str, tuple[str, object]] = {
    "P": ("Pause/Resume", _action_pause_resume),
    "D": ("Delete",       _action_delete),
    "T": ("Retry",        _action_retry),
}
