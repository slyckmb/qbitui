#!/usr/bin/env python3
"""Interactive qBittorrent dashboard with modes, hotkeys, and paging."""
import argparse
import json
import os
import shlex
import select
import shutil
import sys
import readline  # Enables line editing for input()
import subprocess
import termios
import time
import tty
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - optional dependency
    ZoneInfo = None
from pathlib import Path
from http.cookiejar import CookieJar

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

SCRIPT_NAME = "qbit-dashboard"
VERSION = "1.0.0"
LAST_UPDATED = "2026-01-21"

COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_PINK = "\033[38;5;210m"
COLOR_BROWN = "\033[38;5;94m"
COLOR_GREY = "\033[90m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"

LOCAL_TZ = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
PRESET_FILE = Path(__file__).parent.parent / "config" / "qbit-filter-presets.yml"
TRACKERS_LIST_URL = "https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_best.txt"
QC_TAG_TOOL = Path(__file__).resolve().parent / "media_qc_tag.py"
QC_LOG_DIR = Path.home() / ".logs" / "media_qc"

STATE_DOWNLOAD = {"downloading", "stalledDL", "queuedDL", "forcedDL", "metaDL"}
STATE_UPLOAD = {"uploading", "stalledUP", "queuedUP", "forcedUP"}
STATE_PAUSED = {"pausedDL", "pausedUP"}
STATE_ERROR = {"error", "missingFiles"}
STATE_CHECKING = {"checkingUP", "checkingDL", "checkingResumeData", "checking"}
STATE_COMPLETED = {"completed"}

STATE_CODE = {
    "allocating": "A",
    "downloading": "D",
    "checkingDL": "CD",
    "forcedDL": "FD",
    "metaDL": "MD",
    "pausedDL": "PD",
    "queuedDL": "QD",
    "stalledDL": "SD",
    "error": "E",
    "missingFiles": "MF",
    "uploading": "U",
    "checkingUP": "CU",
    "forcedUP": "FU",
    "pausedUP": "PU",
    "queuedUP": "QU",
    "stalledUP": "SU",
    "queuedForChecking": "QC",
    "checkingResumeData": "CR",
    "moving": "MV",
}


def get_key() -> str:
    """Get single keypress."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = ""
            if select.select([sys.stdin], [], [], 0.1)[0]:
                seq = sys.stdin.read(1)
                if seq == "[":
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        _ = sys.stdin.read(1)
                        return ""
                if seq == "O":
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        _ = sys.stdin.read(1)
                        return ""
            return ""
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_line(prompt: str) -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        cooked = termios.tcgetattr(fd)
        cooked[3] |= termios.ICANON | termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, cooked)
        return input(prompt)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def terminal_width() -> int:
    try:
        return max(40, shutil.get_terminal_size((100, 20)).columns)
    except Exception:
        return 100


def read_qbit_config(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    if yaml is not None:
        try:
            data = yaml.safe_load(path.read_text()) or {}
            qb = (data.get("downloaders") or {}).get("qbittorrent", {}) or {}
            return qb.get("api_url", "") or "", qb.get("credentials_file", "") or ""
        except Exception:
            pass

    api_url = ""
    creds = ""
    in_downloaders = False
    in_qbit = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_downloaders = line.strip() == "downloaders:"
            in_qbit = False
            continue
        if in_downloaders and line.startswith("  ") and line.strip().endswith(":"):
            in_qbit = line.strip() == "qbittorrent:"
            continue
        if in_downloaders and in_qbit and line.strip().startswith("api_url:"):
            api_url = line.split("api_url:", 1)[1].strip()
        if in_downloaders and in_qbit and line.strip().startswith("credentials_file:"):
            creds = line.split("credentials_file:", 1)[1].strip()
    return api_url, creds


def read_credentials(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    username = ""
    password = ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("QBITTORRENTAPI_USERNAME="):
            username = line.split("=", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("QBITTORRENTAPI_PASSWORD="):
            password = line.split("=", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("QBITTORRENT_USERNAME="):
            username = line.split("=", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("QBITTORRENT_PASSWORD="):
            password = line.split("=", 1)[1].strip().strip('"').strip("'")
    return username, password


def make_opener() -> urllib.request.OpenerDirector:
    jar = CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def qbit_login(opener: urllib.request.OpenerDirector, api_url: str, username: str, password: str) -> bool:
    data = urllib.parse.urlencode({"username": username, "password": password}).encode()
    req = urllib.request.Request(f"{api_url}/api/v2/auth/login", data=data, method="POST")
    try:
        with opener.open(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return False
    return body == "Ok." or body.strip() == ""


def qbit_request(opener: urllib.request.OpenerDirector, api_url: str, method: str, path: str, params: dict | None = None) -> str:
    url = f"{api_url}{path}"
    data = None
    if params:
        encoded = urllib.parse.urlencode(params)
        if method.upper() == "GET":
            url = f"{url}?{encoded}"
        else:
            data = encoded.encode()
    req = urllib.request.Request(url, data=data, method=method.upper())
    try:
        with opener.open(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return f"HTTP {exc.code}: {body}".strip()


def fetch_public_trackers(url: str) -> list[str]:
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    trackers = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        trackers.append(line)
    return trackers


def spawn_media_qc(hash_value: str) -> str:
    if not QC_TAG_TOOL.exists():
        return f"Missing tool ({QC_TAG_TOOL})"
    QC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = QC_LOG_DIR / f"qc_tag_{hash_value[:8]}.log"
    cmd = [sys.executable, str(QC_TAG_TOOL), "--hash", hash_value, "--apply"]
    with log_path.open("a") as handle:
        handle.write(f"\n=== qc-tag-media {hash_value} @ {datetime.now(LOCAL_TZ).isoformat()} ===\n")
        handle.write(f"cmd={' '.join(cmd)}\n")
    try:
        subprocess.Popen(
            cmd,
            stdout=log_path.open("a"),
            stderr=log_path.open("a"),
            start_new_session=True,
        )
    except Exception as exc:
        return f"Failed ({exc})"
    return f"Queued ({log_path})"


def state_group(state: str) -> str:
    s = state or ""
    if s in STATE_ERROR:
        return "error"
    if s in STATE_PAUSED:
        return "paused"
    if s in STATE_DOWNLOAD:
        return "downloading"
    if s in STATE_UPLOAD:
        return "uploading"
    if s in STATE_COMPLETED:
        return "completed"
    if s in STATE_CHECKING:
        return "checking"
    if s.startswith("queued"):
        return "queued"
    return "other"


def status_color(state: str) -> str:
    group = state_group(state)
    if group == "error":
        return COLOR_RED
    if group == "paused":
        return COLOR_CYAN
    if group in {"downloading", "uploading"}:
        return COLOR_GREEN
    if group in {"queued", "checking"}:
        return COLOR_YELLOW
    if group == "completed":
        return COLOR_GREEN
    return COLOR_RESET


def size_str(value: int | float | None) -> str:
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


def speed_str(value: int | float | None) -> str:
    if value is None:
        return "-"
    try:
        value = float(value)
    except Exception:
        return "-"
    if value <= 0:
        return "0"
    return f"{size_str(value)}/s"


def eta_str(value: int | None) -> str:
    if value is None:
        return "-"
    try:
        value = int(value)
    except Exception:
        return "-"
    if value <= 0 or value >= 8640000:
        return "-"
    mins, sec = divmod(value, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m"
    return f"{mins}m"


def added_str(value: int | float | None) -> str:
    if value is None:
        return "-"
    try:
        value = int(value)
    except Exception:
        return "-"
    if value <= 0:
        return "-"
    return datetime.fromtimestamp(value, LOCAL_TZ).strftime("%Y-%m-%d %H:%M")


def format_ts(value: int | float | None) -> str:
    if value is None:
        return "-"
    try:
        value = int(value)
    except Exception:
        return "-"
    if value <= 0:
        return "-"
    return datetime.fromtimestamp(value, LOCAL_TZ).isoformat()


def summary(torrents: list[dict]) -> str:
    counts = {"down": 0, "up": 0, "paused": 0, "error": 0, "completed": 0, "other": 0}
    for t in torrents:
        group = state_group(t.get("state", ""))
        if group == "downloading":
            counts["down"] += 1
        elif group == "uploading":
            counts["up"] += 1
        elif group == "paused":
            counts["paused"] += 1
        elif group == "error":
            counts["error"] += 1
        elif group == "completed":
            counts["completed"] += 1
        else:
            counts["other"] += 1
    return " ".join([f"{k}:{v}" for k, v in counts.items()])


def build_rows(torrents: list[dict]) -> list[dict]:
    rows = []
    for t in torrents:
        progress = t.get("progress")
        if isinstance(progress, (int, float)):
            progress = f"{int(progress * 100)}%"
        elif progress is None:
            progress = "-"
        rows.append({
            "name": t.get("name", ""),
            "state": t.get("state", ""),
            "st": STATE_CODE.get(t.get("state", ""), "?"),
            "progress": progress,
            "size": size_str(t.get("size") or t.get("total_size")),
            "ratio": f"{t.get('ratio', 0):.2f}" if isinstance(t.get("ratio"), (int, float)) else "-",
            "dlspeed": speed_str(t.get("dlspeed")),
            "upspeed": speed_str(t.get("upspeed")),
            "eta": eta_str(t.get("eta")),
            "added": added_str(t.get("added_on")),
            "category": t.get("category") or "-",
            "tags": t.get("tags") or "-",
            "hash": t.get("hash") or "",
            "raw": t,
        })
    return rows


def format_rows(rows: list, page: int, page_size: int) -> tuple[list, int, int]:
    total_pages = max(1, (len(rows) + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    start = page * page_size
    end = min(start + page_size, len(rows))
    return rows[start:end], total_pages, page


def parse_tag_filter(value: str) -> dict | None:
    raw = value.strip()
    if not raw:
        return None
    if "+" in raw:
        tags = [t.strip().lower() for t in raw.split("+") if t.strip()]
        return {"type": "tag", "mode": "and", "tags": tags, "raw": raw}
    if "," in raw:
        tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return {"type": "tag", "mode": "or", "tags": tags, "raw": raw}
    return {"type": "tag", "mode": "or", "tags": [raw.lower()], "raw": raw}


def parse_filter_line(line: str, existing: list[dict]) -> list[dict]:
    tokens = shlex.split(line)
    if not tokens:
        return existing
    updated = [f for f in existing if f.get("type") not in ("text", "category", "tag")]
    updates: dict[str, dict] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue
        if key in ("text", "q", "name"):
            updates["text"] = {"type": "text", "value": value, "enabled": True}
        elif key in ("cat", "category"):
            updates["category"] = {"type": "category", "value": value, "enabled": True}
        elif key in ("tag", "tags"):
            parsed = parse_tag_filter(value)
            if parsed:
                parsed["enabled"] = True
                updates["tag"] = parsed
    updated.extend(updates.values())
    return updated


def summarize_filters(filters: list[dict]) -> str:
    active = [f for f in filters if f.get("enabled", True)]
    if not active:
        return "-"
    parts = []
    for flt in active:
        if flt["type"] == "text":
            parts.append(f"text={flt['value']}")
        elif flt["type"] == "category":
            parts.append(f"cat={flt['value']}")
        elif flt["type"] == "tag":
            parts.append(f"tag={flt.get('raw', '')}")
    return " ".join(parts)


def format_filters_line(filters: list[dict]) -> str:
    if not filters:
        return "Filters: -"
    parts = []
    for flt in filters:
        active = flt.get("enabled", True)
        color = COLOR_PINK if active else ""
        reset = COLOR_RESET if active else ""
        if flt["type"] == "text":
            parts.append(f"text={color}{flt['value']}{reset}")
        elif flt["type"] == "category":
            cat_color = COLOR_BROWN if active else ""
            cat_reset = COLOR_RESET if active else ""
            parts.append(f"cat={cat_color}{flt['value']}{cat_reset}")
        elif flt["type"] == "tag":
            parts.append(f"tag={color}{flt.get('raw', '')}{reset}")
    return "Filters: " + " ".join(parts)


def load_presets(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        if yaml is None:
            return {}
        data = yaml.safe_load(path.read_text()) or {}
        return data.get("slots") or {}
    except Exception:
        return {}


def save_presets(path: Path, slots: dict) -> None:
    if yaml is None:
        return
    payload = {"slots": slots}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def serialize_filters(filters: list[dict]) -> list[dict]:
    out = []
    for flt in filters:
        if flt["type"] == "text":
            out.append({"type": "text", "value": flt["value"], "enabled": flt.get("enabled", True)})
        elif flt["type"] == "category":
            out.append({"type": "category", "value": flt["value"], "enabled": flt.get("enabled", True)})
        elif flt["type"] == "tag":
            out.append({
                "type": "tag",
                "raw": flt.get("raw", ""),
                "mode": flt.get("mode", "or"),
                "tags": flt.get("tags", []),
                "enabled": flt.get("enabled", True),
            })
    return out


def restore_filters(items: list[dict]) -> list[dict]:
    filters = []
    for item in items or []:
        ftype = item.get("type")
        if ftype == "text" and item.get("value"):
            filters.append({"type": "text", "value": item["value"], "enabled": item.get("enabled", True)})
        elif ftype == "category" and item.get("value"):
            filters.append({"type": "category", "value": item["value"], "enabled": item.get("enabled", True)})
        elif ftype == "tag":
            raw = item.get("raw", "")
            if raw:
                parsed = parse_tag_filter(raw) or {}
                parsed.update({
                    "raw": raw,
                    "mode": item.get("mode", parsed.get("mode", "or")),
                    "tags": item.get("tags", parsed.get("tags", [])),
                    "enabled": item.get("enabled", True),
                })
                filters.append(parsed)
    return filters


def apply_filters(rows: list[dict], filters: list[dict]) -> list[dict]:
    active = [f for f in filters if f.get("enabled", True)]
    if not active:
        return rows
    filtered = rows
    for flt in active:
        if flt["type"] == "text":
            term = flt["value"].lower()
            filtered = [r for r in filtered if term in (r.get("name") or "").lower()]
        elif flt["type"] == "category":
            category = flt["value"].lower()
            if category == "-":
                filtered = [r for r in filtered if not (r.get("raw", {}).get("category") or "").strip()]
            else:
                filtered = [r for r in filtered if (r.get("category") or "").lower() == category]
        elif flt["type"] == "tag":
            tags = set(flt.get("tags") or [])
            mode = flt.get("mode", "or")
            if not tags:
                continue
            def match(row: dict) -> bool:
                raw_tags = row.get("raw", {}).get("tags") or ""
                tag_set = {t.strip().lower() for t in raw_tags.split(",") if t.strip()}
                if mode == "and":
                    return tags.issubset(tag_set)
                return bool(tags & tag_set)
            filtered = [r for r in filtered if match(r)]
    return filtered


def apply_action(opener: urllib.request.OpenerDirector, api_url: str, mode: str, item: dict) -> str:
    hash_value = item.get("hash")
    if not hash_value:
        return "Missing hash"
    state = item.get("state") or ""
    raw = item.get("raw", {})

    if mode == "p":
        action = "resume" if "paused" in state.lower() else "pause"
        qbit_request(opener, api_url, "POST", f"/api/v2/torrents/{action}", {"hashes": hash_value})
        return "OK"
    if mode == "d":
        delete_files = read_line("Delete files too? (y/N): ").strip().lower() == "y"
        qbit_request(opener, api_url, "POST", "/api/v2/torrents/delete", {"hashes": hash_value, "deleteFiles": "true" if delete_files else "false"})
        return "OK"
    if mode == "c":
        value = read_line("Enter new category (blank cancels): ").strip()
        if not value:
            return "Cancelled"
        qbit_request(opener, api_url, "POST", "/api/v2/torrents/setCategory", {"hashes": hash_value, "category": value})
        return "OK"
    if mode == "t":
        existing_tags = (item.get("tags") or "").strip()
        if existing_tags:
            print(f"Current tags: {existing_tags}")
        value = read_line("Tags (comma-separated, '-' to remove, '--' to clear all, blank cancels): ").strip()
        if not value:
            return "Cancelled"
        if value == "--":
            existing = (item.get("tags") or "").strip()
            if not existing:
                return "No tags to clear"
            qbit_request(opener, api_url, "POST", "/api/v2/torrents/removeTags", {"hashes": hash_value, "tags": existing})
            return "OK"
        if value.startswith("-"):
            tags = value[1:].strip()
            if not tags:
                return "Cancelled"
            qbit_request(opener, api_url, "POST", "/api/v2/torrents/removeTags", {"hashes": hash_value, "tags": tags})
            return "OK"
        qbit_request(opener, api_url, "POST", "/api/v2/torrents/addTags", {"hashes": hash_value, "tags": value})
        return "OK"
    if mode == "v":
        qbit_request(opener, api_url, "POST", "/api/v2/torrents/recheck", {"hashes": hash_value})
        return "OK"
    if mode == "A":
        priv = raw.get("private")
        if priv is None:
            return "Skip (private=unknown)"
        if isinstance(priv, str):
            priv = priv.strip().lower()
            if priv in ("true", "1", "yes"):
                return "Skip (private)"
            if priv in ("false", "0", "no"):
                priv = False
        if priv:
            return "Skip (private)"
        trackers = fetch_public_trackers(TRACKERS_LIST_URL)
        if not trackers:
            return "Failed (no trackers)"
        urls = "\n".join(trackers)
        resp = qbit_request(opener, api_url, "POST", "/api/v2/torrents/addTrackers", {"hash": hash_value, "urls": urls})
        if resp.startswith("HTTP "):
            return f"Failed ({resp})"
        return f"OK ({len(trackers)})"
    if mode == "Q":
        return spawn_media_qc(hash_value)
    return "Unknown mode"


def print_details(item: dict) -> None:
    raw = item.get("raw") or {}
    print(f"{COLOR_BOLD}Details{COLOR_RESET}")
    print(f"  Name: {item.get('name')}")
    print(f"  State: {item.get('state')}")
    print(f"  Category: {item.get('category')}")
    print(f"  Tags: {item.get('tags')}")
    print(f"  Size: {item.get('size')}")
    print(f"  Progress: {item.get('progress')}")
    print(f"  Ratio: {item.get('ratio')}")
    print(f"  DL/UL: {item.get('dlspeed')} / {item.get('upspeed')}")
    print(f"  ETA: {item.get('eta')}")
    print(f"  Hash: {item.get('hash')}")
    for key in ("save_path", "content_path", "tracker", "completion_on", "added_on", "last_activity", "state", "magnet_uri"):
        if key in raw:
            value = raw.get(key)
            if key.endswith("_on") and isinstance(value, (int, float)):
                value = format_ts(value)
            print(f"  {key}: {value}")
    print("")
    print("Press any key to continue...", end="", flush=True)
    _ = get_key()


def print_raw(item: dict) -> None:
    raw = item.get("raw") or {}
    print(f"{COLOR_BOLD}Raw JSON{COLOR_RESET}")
    print(json.dumps(raw, indent=2, sort_keys=True))
    print("")
    print("Press any key to continue...", end="", flush=True)
    _ = get_key()


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive qBittorrent dashboard")
    parser.add_argument("--config", default=os.environ.get("QBITTORRENT_CONFIG_FILE"), help="Path to request-cache.yml")
    parser.add_argument("--page-size", type=int, default=int(os.environ.get("QBITTORRENT_PAGE_SIZE", "10")))
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else (Path(__file__).parent.parent / "config" / "request-cache.yml")
    cfg_api_url, cfg_creds = read_qbit_config(config_path)
    api_url = os.environ.get("QBITTORRENT_API_URL") or cfg_api_url or "http://localhost:9003"
    creds_file = os.environ.get("QBITTORRENT_CREDENTIALS_FILE") or cfg_creds or "/mnt/config/secrets/qbittorrent/api.env"

    username = os.environ.get("QBITTORRENT_USERNAME", "")
    password = os.environ.get("QBITTORRENT_PASSWORD", "")
    if not username or not password:
        username, password = read_credentials(Path(creds_file))
    if not username or not password:
        print("ERROR: QBITTORRENT credentials not found (set env or credentials file)", file=sys.stderr)
        return 1

    opener = make_opener()
    if not qbit_login(opener, api_url, username, password):
        print("ERROR: qBittorrent login failed", file=sys.stderr)
        return 1

    mode = "i"
    scope = "all"
    page = 0
    filters: list[dict] = []
    presets = load_presets(PRESET_FILE)
    sort_fields = ["added_on", "name", "state", "ratio", "progress", "eta", "size", "dlspeed", "upspeed"]
    sort_index = 0
    sort_desc = True
    show_tags = False
    show_full_hash = False

    while True:
        raw = qbit_request(opener, api_url, "GET", "/api/v2/torrents/info")
        try:
            torrents = json.loads(raw) if raw else []
        except json.JSONDecodeError:
            torrents = []

        rows = build_rows(torrents)

        if scope != "all":
            rows = [r for r in rows if state_group(r.get("raw", {}).get("state", "")) == scope]

        rows = apply_filters(rows, filters)

        sort_field = sort_fields[sort_index]
        def sort_key(row: dict):
            raw = row.get("raw", {})
            if sort_field == "added_on":
                return raw.get("added_on") or 0
            if sort_field == "name":
                return row.get("name", "")
            if sort_field == "state":
                return state_group(raw.get("state", ""))
            if sort_field == "ratio":
                return raw.get("ratio") or 0
            if sort_field == "progress":
                return raw.get("progress") or 0
            if sort_field == "eta":
                return raw.get("eta") or 0
            if sort_field == "size":
                return raw.get("size") or raw.get("total_size") or 0
            if sort_field == "dlspeed":
                return raw.get("dlspeed") or 0
            if sort_field == "upspeed":
                return raw.get("upspeed") or 0
            return row.get("name", "")
        rows.sort(key=sort_key, reverse=sort_desc)
        page_rows, total_pages, page = format_rows(rows, page, args.page_size)

        os.system("clear")
        print(f"{COLOR_BOLD}QBITTORRENT DASHBOARD (TUI){COLOR_RESET}")
        print(f"API: {api_url}")
        print(f"Summary: {summary(torrents)}")
        print("")

        scope_label = scope.upper()
        mode_label = {"i": "INFO", "p": "PAUSE/RESUME", "d": "DELETE", "c": "CATEGORY", "t": "TAGS", "v": "VERIFY", "A": "ADD PUBLIC TRACKERS", "Q": "QC TAG MEDIA"}[mode]
        page_label = f"Page {page + 1}/{total_pages}"
        sort_label = f"{sort_field} ({'desc' if sort_desc else 'asc'})"
        print(f"Mode: {COLOR_BLUE}{mode_label}{COLOR_RESET}  Scope: {COLOR_MAGENTA}{scope_label}{COLOR_RESET}  Sort: {COLOR_GREY}{sort_label}{COLOR_RESET}  {page_label}")

        print("")
        hash_width = 40 if show_full_hash else 6
        hash_label = "Hash"
        header_line = f"{'No':<3} {'ST':<2} {'Name':<44} {'Prog':<6} {'Size':<10} {'DL':<8} {'UL':<8} {'ETA':<6} {'Added':<16} {'Cat':<10} {hash_label:<{hash_width}}"
        divider_line = "-" * min(len(header_line), terminal_width())
        print(header_line)
        print(divider_line)

        for idx, item in enumerate(page_rows, 0):
            color = status_color(item.get("state") or "")
            name = (item.get("name") or "")[:44]
            state = item.get("state") or ""
            st = item.get("st") or "?"
            cat_val = str(item.get("category") or "-")
            base_line = (
                f"{idx:<3} {color}{st:<2}{COLOR_RESET} "
                f"{color}{name:<44}{COLOR_RESET} "
                f"{str(item.get('progress') or '-'): <6} "
                f"{str(item.get('size') or '-'): <10} "
                f"{str(item.get('dlspeed') or '-'): <8} "
                f"{str(item.get('upspeed') or '-'): <8} "
                f"{str(item.get('eta') or '-'): <6} "
                f"{str(item.get('added') or '-'): <16} "
                f"{COLOR_BROWN}{cat_val:<10}{COLOR_RESET} "
            )
            hash_value = str(item.get("hash") or "")
            hash_display = hash_value if show_full_hash else hash_value[:6] or "-"
            print(f"{base_line}{hash_display:<{hash_width}}")
            if show_tags:
                tags_raw = str(item.get("tags") or "").strip()
                if tags_raw:
                    tag_parts = []
                    for tag in [t.strip() for t in tags_raw.split(",") if t.strip()]:
                        if "FAIL" in tag.upper():
                            tag_parts.append(f"{COLOR_RED}{tag}{COLOR_RESET}")
                        else:
                            tag_parts.append(f"{COLOR_PINK}{tag}{COLOR_RESET}")
                    tags_line = ", ".join(tag_parts)
                    print(f"     tags: {tags_line}")

        print(divider_line)
        print(format_filters_line(filters))
        print(divider_line)
        print(
            "Keys: 0-9=Apply  r=Refresh  D=Default  f=Filter  a=All  w=Down  u=Up  z=Paused  e=Done  g=Err  s=Sort  H=Hash"
        )
        print(
            "      C=Cat  #=Tag  /=Line  F=Filters  P=Presets  S=Dir  T=Tags  [=Prev ]=Next  i/p/d/c/t/v/A/Q=Mode  R=Raw  ?=Help  x=Quit"
        )
        print(divider_line)

        key = get_key()
        if key in ("x", "\x1b"):
            break
        if key == "?":
            print("Modes: i=info, p=pause/resume, d=delete, c=category, t=tags, v=verify, A=add public trackers (non-private), Q=qc-tag-media")
            print("Paging: ] next page, [ previous page")
            print("Scope: a=all, w=downloading, u=uploading, z=paused, e=completed, g=error  Raw: R + item number")
            print("Sort: s=cycle field, S=toggle asc/desc  Columns: T=toggle tags, H=toggle hash width")
            print("Filters: f=text, C=category, #=tag (comma=OR, plus=AND), /=line, F=manage stack, P=presets")
            print("Press any key to continue...", end="", flush=True)
            _ = get_key()
            continue
        if key == "D":
            mode = "i"
            scope = "all"
            page = 0
            filters = []
            sort_index = 0
            sort_desc = True
            show_tags = False
            continue
        if key == "r":
            continue
        if key == "a":
            scope = "all"
            page = 0
            continue
        if key == "w":
            scope = "downloading"
            page = 0
            continue
        if key == "u":
            scope = "uploading"
            page = 0
            continue
        if key == "z":
            scope = "paused"
            page = 0
            continue
        if key == "e":
            scope = "completed"
            page = 0
            continue
        if key == "g":
            scope = "error"
            page = 0
            continue
        if key == "f":
            value = read_line("\nText filter (blank clears): ").strip()
            filters = [f for f in filters if f["type"] != "text"]
            if value:
                filters.append({"type": "text", "value": value, "enabled": True})
            page = 0
            continue
        if key == "/":
            line = read_line("\nFilter line (text=... cat=... tag=...): ").strip()
            if line:
                filters = parse_filter_line(line, filters)
                page = 0
            continue
        if key == "C":
            value = read_line("\nCategory filter (blank clears): ").strip()
            filters = [f for f in filters if f["type"] != "category"]
            if value:
                filters.append({"type": "category", "value": value, "enabled": True})
            page = 0
            continue
        if key == "#":
            value = read_line("\nTag filter (comma=OR, plus=AND; blank clears): ").strip()
            filters = [f for f in filters if f["type"] != "tag"]
            parsed = parse_tag_filter(value)
            if parsed:
                parsed["enabled"] = True
                filters.append(parsed)
            page = 0
            continue
        if key == "P":
            print("\nFilter presets:")
            if not presets:
                print("  (none)")
            else:
                for slot in sorted(presets.keys()):
                    slot_data = presets[slot] or {}
                    name = slot_data.get("name") or ""
                    items = restore_filters(slot_data.get("filters") or [])
                    label = summarize_filters(items)
                    print(f"  {slot}: {name} {label}")
            choice = read_line("Enter slot to load, sN to save, rN to remove (blank cancels): ").strip()
            if not choice:
                continue
            if choice.startswith("s") and choice[1:].isdigit():
                slot = choice[1:]
                name = read_line(f"Slot {slot} name (blank keeps): ").strip()
                existing = presets.get(slot, {})
                presets[slot] = {
                    "name": name or existing.get("name", ""),
                    "filters": serialize_filters(filters),
                }
                save_presets(PRESET_FILE, presets)
                continue
            if choice.startswith("r") and choice[1:].isdigit():
                slot = choice[1:]
                if slot in presets:
                    del presets[slot]
                    save_presets(PRESET_FILE, presets)
                continue
            if choice.isdigit() and choice in presets:
                filters = restore_filters(presets[choice].get("filters") or [])
                page = 0
            continue
        if key == "F":
            if not filters:
                print("\nNo filters set. Press any key to continue...", end="", flush=True)
                _ = get_key()
                continue
            print("\nActive filters:")
            for idx, flt in enumerate(filters, 1):
                flag = "x" if flt.get("enabled", True) else " "
                if flt["type"] == "text":
                    label = f"text={flt['value']}"
                elif flt["type"] == "category":
                    label = f"cat={flt['value']}"
                else:
                    label = f"tag={flt.get('raw', '')}"
                print(f"  {idx}. [{flag}] {label}")
            choice = read_line("Enter number to toggle, rN to remove, 0 clears all (blank cancels): ").strip()
            if choice == "0":
                filters = []
            elif choice.startswith("r") and choice[1:].isdigit():
                idx = int(choice[1:])
                if 1 <= idx <= len(filters):
                    del filters[idx - 1]
            elif choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(filters):
                    filters[idx - 1]["enabled"] = not filters[idx - 1].get("enabled", True)
            page = 0
            continue
        if key == "s":
            sort_index = (sort_index + 1) % len(sort_fields)
            page = 0
            continue
        if key == "S":
            sort_desc = not sort_desc
            page = 0
            continue
        if key == "T":
            show_tags = not show_tags
            continue
        if key == "H":
            show_full_hash = not show_full_hash
            continue
        if key in "ipdctvAQ":
            mode = key
            continue
        if key == "[":
            page = total_pages - 1 if page == 0 else page - 1
            continue
        if key == "]":
            page = 0 if page >= total_pages - 1 else page + 1
            continue
        if key == "R":
            raw_choice = read_line("\nRaw item number (blank cancels): ").strip()
            if raw_choice.isdigit():
                idx = int(raw_choice)
                if 1 <= idx <= len(page_rows):
                    print("")
                    print_raw(page_rows[idx - 1])
            continue
        if key and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(page_rows):
                item = page_rows[idx]
                if mode == "i":
                    print("")
                    print_details(item)
                else:
                    result = apply_action(opener, api_url, mode, item)
                    print(f"{mode_label}: {result}")
                    time.sleep(0.6)
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
