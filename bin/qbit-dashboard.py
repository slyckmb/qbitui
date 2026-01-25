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
import re
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
from typing import Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

SCRIPT_NAME = "qbit-dashboard"
VERSION = "1.1.0"
LAST_UPDATED = "2026-01-21"

COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_PINK = "\033[38;5;210m"
COLOR_ORANGE = "\033[38;5;214m"
COLOR_BRIGHT_GREEN = "\033[92m"
COLOR_BRIGHT_CYAN = "\033[96m"
COLOR_BRIGHT_YELLOW = "\033[93m"
COLOR_BRIGHT_BLUE = "\033[94m"
COLOR_BRIGHT_PURPLE = "\033[95m"
COLOR_BRIGHT_RED = "\033[91m"
COLOR_BRIGHT_WHITE = "\033[97m"
COLOR_BROWN = "\033[38;5;214m"
COLOR_GREY = "\033[38;5;245m"
COLOR_BOLD = "\033[1m"
COLOR_DEFAULT = "\033[38;5;250m"
COLOR_RESET = "\033[0m\033[38;5;250m"

LOCAL_TZ = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
PRESET_FILE = Path(__file__).parent.parent / "config" / "qbit-filter-presets.yml"
TRACKERS_LIST_URL = "https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_best.txt"
QC_TAG_TOOL = Path(__file__).resolve().parent / "media_qc_tag.py"
QC_LOG_DIR = Path.home() / ".logs" / "media_qc"

STATE_DOWNLOAD = {"downloading", "stalledDL", "queuedDL", "forcedDL", "metaDL"}
STATE_UPLOAD = {"uploading", "stalledUP", "queuedUP", "forcedUP"}
STATE_PAUSED = {"pausedDL", "pausedUP", "stoppedDL", "stoppedUP"}
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
    "stoppedDL": "PD",
    "error": "E",
    "missingFiles": "MF",
    "uploading": "U",
    "checkingUP": "CU",
    "forcedUP": "FU",
    "pausedUP": "PU",
    "queuedUP": "QU",
    "stalledUP": "SU",
    "stoppedUP": "PU",
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
    s = (state or "").strip()
    if s in {"error", "missingFiles"}:
        return COLOR_BRIGHT_RED
    if s in {"downloading", "forcedDL", "metaDL"}:
        return COLOR_BRIGHT_GREEN
    if s in {"uploading", "forcedUP"}:
        return COLOR_CYAN
    if s in {"pausedDL", "pausedUP"}:
        return COLOR_BRIGHT_YELLOW
    if s in {"completed"}:
        return COLOR_BRIGHT_PURPLE
    if s in {"queuedDL", "queuedUP", "checkingUP", "checkingDL", "checkingResumeData", "queuedForChecking", "checking"}:
        return COLOR_BRIGHT_BLUE
    if s in {"stalledUP", "stalledDL", "allocating", "moving"}:
        return COLOR_ORANGE
    if not s:
        return COLOR_BRIGHT_WHITE
    return COLOR_BRIGHT_WHITE


def mode_color(mode: str) -> str:
    colors = {
        "i": COLOR_BRIGHT_CYAN,
        "p": COLOR_BRIGHT_YELLOW,
        "d": COLOR_BRIGHT_RED,
        "c": COLOR_ORANGE,
        "t": COLOR_PINK,
        "v": COLOR_BRIGHT_BLUE,
        "A": COLOR_BRIGHT_GREEN,
        "Q": COLOR_BRIGHT_PURPLE,
        "l": COLOR_BRIGHT_WHITE,
        "m": COLOR_MAGENTA,
    }
    return colors.get(mode, COLOR_RESET)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(value: str) -> int:
    return len(ANSI_RE.sub("", value))


def wrap_ansi(value: str, width: int) -> list[str]:
    if width <= 0:
        return [value]
    lines = []
    current = ""
    for chunk in value.split(" "):
        if not current:
            current = chunk
            continue
        if visible_len(current) + 1 + visible_len(chunk) <= width:
            current = current + " " + chunk
        else:
            lines.append(current)
            current = chunk
    if current:
        lines.append(current)
    return lines


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


def truncate(value: str, max_len: int) -> str:
    if len(value) <= max_len:
        return value
    if max_len <= 1:
        return value[:max_len]
    if max_len <= 3:
        return value[:max_len]
    return value[: max_len - 3] + "..."


CACHE_DIR = Path(__file__).parent.parent / "cache" / "mediainfo"


def get_largest_media_file(content_path: str) -> Optional[Path]:
    if not content_path:
        return None
    path = Path(content_path)
    if not path.exists():
        return None
    
    exts = {".mkv", ".mp4", ".avi", ".m4v", ".mov", ".ts", ".m2ts", ".mpg", ".mpeg", ".webm", ".wmv",
            ".mp3", ".m4b", ".m4a", ".flac", ".aac", ".ogg", ".wav"}
    
    files = []
    if path.is_file():
        if path.suffix.lower() in exts:
            files.append(path)
    else:
        for item_path in path.rglob("*"):
            if item_path.is_file() and item_path.suffix.lower() in exts:
                files.append(item_path)
    
    if not files:
        return None
    
    # Sort by size descending
    files.sort(key=lambda x: x.stat().st_size, reverse=True)
    return files[0]


def get_mediainfo_summary(hash_value: str, content_path: str) -> str:
    if not hash_value:
        return "ERROR: Missing hash"
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{hash_value}.summary"

    if cache_file.exists():
        return cache_file.read_text().strip()

    target = get_largest_media_file(content_path)
    if not target:
        mi_summary = "No media content."
        cache_file.write_text(mi_summary)
        return mi_summary

    tool = shutil.which("mediainfo")
    if not tool:
        return "ERROR: mediainfo not found"

    # Concise summary format: Resolution | VideoCodec | BitRate | AudioCodec | Channels
    inform = "Video;[%Width%x%Height%] [%Format%] [%BitRate/String%]|Audio; [%Format%] [%Channel(s)%ch]"
    
    result = subprocess.run(
        [tool, f"--Inform={inform}", str(target)],
        capture_output=True,
        text=True,
    )
    
    mi_summary = (result.stdout or "").strip()
    if not mi_summary:
        mi_summary = "MediaInfo extraction failed."
    
    # Clean up and normalize
    mi_summary = mi_summary.replace("][", "] [").replace("  ", " ").strip()
    if mi_summary.startswith("|"):
        mi_summary = mi_summary[1:].strip()
    
    cache_file.write_text(mi_summary)
    return mi_summary


def get_mediainfo_for_hash(hash_value: str, content_path: str) -> str:
    if not hash_value:
        return "ERROR: Missing hash"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{hash_value}.txt"

    if cache_file.exists():
        return cache_file.read_text()

    if not content_path:
        return "ERROR: No content path"
    
    path = Path(content_path)
    if not path.exists():
        return f"ERROR: Path not found ({path})"

    files = []
    if path.is_file():
        files = [path]
    else:
        exts = {".mkv", ".mp4", ".avi", ".m4v", ".mov", ".ts", ".m2ts", ".mpg", ".mpeg", ".webm", ".wmv",
                ".mp3", ".m4b", ".m4a", ".flac", ".aac", ".ogg", ".wav"}
        for item_path in sorted(path.rglob("*")):
            if item_path.is_file() and item_path.suffix.lower() in exts:
                files.append(item_path)
    
    if not files:
        return "ERROR: No media files found"

    table = mediainfo_table(files)
    cache_file.write_text(table)
    return table


def mediainfo_table(paths: list[Path]) -> str:
    tool = shutil.which("mediainfo")
    if not tool:
        return "ERROR: mediainfo not found"
    inform = (
        "General;%FileName%|%Duration/String3%|%FileSize/String%|%OverallBitRate/String%|"
        "%Format%|%Width%x%Height%|%FrameRate%|%BitRate/String%|%Channel(s)%|%SamplingRate/String%|%BitRate/String%\\n"
    )
    headers = [
        "FileName", "Duration", "FileSize", "OverallBitRate", "Format",
        "WxH", "FrameRate", "VideoBitRate", "Channels", "SamplingRate", "AudioBitRate",
    ]
    rows = []
    for path in paths:
        result = subprocess.run(
            [tool, f"--Inform={inform}", str(path)],
            capture_output=True,
            text=True,
        )
        line = (result.stdout or "").strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < len(headers):
            parts += [""] * (len(headers) - len(parts))
        rows.append(parts[: len(headers)])
    if not rows:
        return "No mediainfo output"

    caps = [60, 12, 10, 12, 10, 9, 8, 12, 8, 12, 12]
    widths = []
    for idx, header in enumerate(headers):
        max_len = max(len(header), max(len(r[idx]) for r in rows))
        widths.append(min(max_len, caps[idx]))

    lines = []
    lines.append("  ".join(truncate(headers[i], widths[i]).ljust(widths[i]) for i in range(len(headers))))
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        line = "  ".join(truncate(str(row[i]), widths[i]).ljust(widths[i]) for i in range(len(headers)))
        lines.append(line)
    return "\n".join(lines)


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
    expr = parse_tag_expr(raw)
    if expr:
        return {"type": "tag", "raw": raw, "expr": expr}
    if "+" in raw:
        tags = [t.strip().lower() for t in raw.split("+") if t.strip()]
        return {"type": "tag", "mode": "and", "tags": tags, "raw": raw}
    if "," in raw:
        tags = [t.strip().lower() for t in raw.split(",") if t.strip()]
        return {"type": "tag", "mode": "or", "tags": tags, "raw": raw}
    return {"type": "tag", "mode": "or", "tags": [raw.lower()], "raw": raw}


def _tokenize_tag_expr(value: str) -> list[str]:
    tokens = []
    i = 0
    ops = set("+,()!")
    while i < len(value):
        ch = value[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ops:
            tokens.append(ch)
            i += 1
            continue
        start = i
        while i < len(value) and not value[i].isspace() and value[i] not in ops:
            i += 1
        tokens.append(value[start:i])
    return tokens


def parse_tag_expr(value: str):
    tokens = _tokenize_tag_expr(value)
    if not tokens:
        return None
    idx = 0

    def parse_expr():
        return parse_or()

    def parse_or():
        node = parse_and()
        items = [node]
        while current() == ",":
            advance()
            items.append(parse_and())
        if len(items) == 1:
            return items[0]
        return ("or", items)

    def parse_and():
        node = parse_unary()
        items = [node]
        while current() == "+":
            advance()
            items.append(parse_unary())
        if len(items) == 1:
            return items[0]
        return ("and", items)

    def parse_unary():
        if current() == "!":
            advance()
            return ("not", parse_unary())
        if current() == "(":
            advance()
            node = parse_expr()
            if current() != ")":
                return None
            advance()
            return node
        token = current()
        if token in (None, "+", ",", ")", "!"):
            return None
        advance()
        return ("tag", token.lower())

    def current():
        return tokens[idx] if idx < len(tokens) else None

    def advance():
        nonlocal idx
        idx += 1

    tree = parse_expr()
    if tree is None:
        return None
    if idx != len(tokens):
        return None
    return tree


def eval_tag_expr(node, tag_set: set[str]) -> bool:
    kind = node[0]
    if kind == "tag":
        return node[1] in tag_set
    if kind == "not":
        return not eval_tag_expr(node[1], tag_set)
    if kind == "and":
        return all(eval_tag_expr(item, tag_set) for item in node[1])
    if kind == "or":
        return any(eval_tag_expr(item, tag_set) for item in node[1])
    return False

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
            negate = False
            if value.startswith("!"):
                negate = True
                value = value[1:]
            updates["text"] = {"type": "text", "value": value, "enabled": True, "negate": negate}
        elif key in ("cat", "category"):
            negate = False
            if value.startswith("!"):
                negate = True
                value = value[1:]
            updates["category"] = {"type": "category", "value": value, "enabled": True, "negate": negate}
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
        prefix = "!" if flt.get("negate") else ""
        if flt["type"] == "text":
            parts.append(f"text={prefix}{flt['value']}")
        elif flt["type"] == "category":
            parts.append(f"cat={prefix}{flt['value']}")
        elif flt["type"] == "tag":
            raw = flt.get("raw", "")
            if prefix and raw and not raw.startswith("!"):
                raw = prefix + raw
            parts.append(f"tag={raw}")
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
            prefix = "!" if flt.get("negate") else ""
            parts.append(f"text={color}{prefix}{flt['value']}{reset}")
        elif flt["type"] == "category":
            cat_color = COLOR_BROWN if active else ""
            cat_reset = COLOR_RESET if active else ""
            prefix = "!" if flt.get("negate") else ""
            parts.append(f"cat={cat_color}{prefix}{flt['value']}{cat_reset}")
        elif flt["type"] == "tag":
            raw = flt.get("raw", "")
            if flt.get("negate") and raw and not raw.startswith("!"):
                raw = "!" + raw
            parts.append(f"tag={color}{raw}{reset}")
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
            out.append({"type": "text", "value": flt["value"], "enabled": flt.get("enabled", True), "negate": flt.get("negate", False)})
        elif flt["type"] == "category":
            out.append({"type": "category", "value": flt["value"], "enabled": flt.get("enabled", True), "negate": flt.get("negate", False)})
        elif flt["type"] == "tag":
            out.append({
                "type": "tag",
                "raw": flt.get("raw", ""),
                "mode": flt.get("mode", "or"),
                "tags": flt.get("tags", []),
                "enabled": flt.get("enabled", True),
                "negate": flt.get("negate", False),
            })
    return out


def restore_filters(items: list[dict]) -> list[dict]:
    filters = []
    for item in items or []:
        ftype = item.get("type")
        if ftype == "text" and item.get("value"):
            filters.append({"type": "text", "value": item["value"], "enabled": item.get("enabled", True), "negate": item.get("negate", False)})
        elif ftype == "category" and item.get("value"):
            filters.append({"type": "category", "value": item["value"], "enabled": item.get("enabled", True), "negate": item.get("negate", False)})
        elif ftype == "tag":
            raw = item.get("raw", "")
            if raw:
                parsed = parse_tag_filter(raw) or {}
                parsed.update({
                    "raw": raw,
                    "mode": item.get("mode", parsed.get("mode", "or")),
                    "tags": item.get("tags", parsed.get("tags", [])),
                    "enabled": item.get("enabled", True),
                    "negate": item.get("negate", False),
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
            def match_text(r):
                present = term in (r.get("name") or "").lower()
                return not present if flt.get("negate") else present
            filtered = [r for r in filtered if match_text(r)]
        elif flt["type"] == "category":
            category = flt["value"].lower()
            def match_cat(r):
                raw_cat = (r.get("raw", {}).get("category") or "").strip().lower()
                if category == "-":
                    present = not raw_cat
                else:
                    present = raw_cat == category
                return not present if flt.get("negate") else present
            filtered = [r for r in filtered if match_cat(r)]
        elif flt["type"] == "tag":
            tags = set(flt.get("tags") or [])
            mode = flt.get("mode", "or")
            expr = flt.get("expr")
            if not tags and not expr:
                continue
            def match(row: dict) -> bool:
                raw_tags = row.get("raw", {}).get("tags") or ""
                tag_set = {t.strip().lower() for t in raw_tags.split(",") if t.strip()}
                if expr:
                    present = eval_tag_expr(expr, tag_set)
                elif mode == "and":
                    present = tags.issubset(tag_set)
                else:
                    present = bool(tags & tag_set)
                return not present if flt.get("negate") else present
            filtered = [r for r in filtered if match(r)]
    return filtered


def apply_action(opener: urllib.request.OpenerDirector, api_url: str, mode: str, item: dict) -> str:
    hash_value = item.get("hash")
    if not hash_value:
        return "Missing hash"
    state = item.get("state") or ""
    raw = item.get("raw", {})

    if mode == "p":
        is_paused = "paused" in state.lower() or "stopped" in state.lower()
        action = "start" if is_paused else "stop"
        resp = qbit_request(opener, api_url, "POST", f"/api/v2/torrents/{action}", {"hashes": hash_value})
        # Try fallbacks for older versions if start/stop 404
        if "HTTP 404" in resp:
            old_action = "resume" if is_paused else "pause"
            resp = qbit_request(opener, api_url, "POST", f"/api/v2/torrents/{old_action}", {"hashes": hash_value})
        return "OK" if resp in ("Ok.", "") else resp
    if mode == "d":
        delete_files = read_line("Delete files too? (y/N): ").strip().lower() == "y"
        resp = qbit_request(opener, api_url, "POST", "/api/v2/torrents/delete", {"hashes": hash_value, "deleteFiles": "true" if delete_files else "false"})
        return "OK" if resp in ("Ok.", "") else resp
    if mode == "c":
        value = read_line("Enter new category (blank cancels): ").strip()
        if not value:
            return "Cancelled"
        resp = qbit_request(opener, api_url, "POST", "/api/v2/torrents/setCategory", {"hashes": hash_value, "category": value})
        return "OK" if resp in ("Ok.", "") else resp
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
            resp = qbit_request(opener, api_url, "POST", "/api/v2/torrents/removeTags", {"hashes": hash_value, "tags": existing})
            return "OK" if resp in ("Ok.", "") else resp
        if value.startswith("-"):
            tags = value[1:].strip()
            if not tags:
                return "Cancelled"
            resp = qbit_request(opener, api_url, "POST", "/api/v2/torrents/removeTags", {"hashes": hash_value, "tags": tags})
            return "OK" if resp in ("Ok.", "") else resp
        resp = qbit_request(opener, api_url, "POST", "/api/v2/torrents/addTags", {"hashes": hash_value, "tags": value})
        return "OK" if resp in ("Ok.", "") else resp
    if mode == "v":
        resp = qbit_request(opener, api_url, "POST", "/api/v2/torrents/recheck", {"hashes": hash_value})
        return "OK" if resp in ("Ok.", "") else resp
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


def print_files(opener: urllib.request.OpenerDirector, api_url: str, item: dict) -> None:
    hash_value = item.get("hash")
    if not hash_value:
        print("Missing hash")
        _ = get_key()
        return

    raw = qbit_request(opener, api_url, "GET", "/api/v2/torrents/files", {"hash": hash_value})
    try:
        files = json.loads(raw)
    except Exception:
        print(f"Failed to parse files: {raw}")
        _ = get_key()
        return

    if not files:
        print("No files found.")
        _ = get_key()
        return

    print(f"{COLOR_BOLD}Files for: {item.get('name')}{COLOR_RESET}")
    headers = ["Index", "Name", "Size", "Prog", "Priority"]
    widths = [5, 60, 10, 6, 10]
    
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))

    # Sort files by name
    files.sort(key=lambda x: x.get("name", ""))
    
    priority_map = {0: "Do not DL", 1: "Normal", 2: "High", 6: "Max", 7: "Forced"}

    for idx, f in enumerate(files):
        name = truncate(f.get("name", ""), widths[1])
        size = size_str(f.get("size", 0))
        prog = f"{int(f.get('progress', 0) * 100)}%"
        prio = priority_map.get(f.get("priority", 1), str(f.get("priority")))
        
        line = (
            f"{str(idx):<5} "
            f"{name:<60} "
            f"{size:<10} "
            f"{prog:<6} "
            f"{prio:<10}"
        )
        print(line)

    print("")
    print("Press any key to continue...", end="", flush=True)
    _ = get_key()


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


def print_mediainfo(item: dict) -> None:
    raw = item.get("raw") or {}
    content_path = raw.get("content_path")
    if not content_path:
        save_path = raw.get("save_path") or ""
        name = raw.get("name") or ""
        content_path = str(Path(save_path) / name) if save_path and name else ""
    
    info = get_mediainfo_for_hash(item.get("hash"), content_path)
    print(f"{COLOR_BOLD}MediaInfo for: {item.get('name')}{COLOR_RESET}")
    print(info)
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
    show_mediainfo_inline = False
    show_full_hash = False
    show_added = True

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

        # Pre-populate MediaInfo cache if toggle is on
        if show_mediainfo_inline:
            to_fetch = []
            for r in page_rows:
                if not (CACHE_DIR / f"{r['hash']}.summary").exists():
                    to_fetch.append(r)
            
            if to_fetch:
                print(f"{COLOR_YELLOW}Populating MediaInfo cache for {len(to_fetch)} items...{COLOR_RESET}")
                for idx, r in enumerate(to_fetch, 1):
                    raw = r.get("raw") or {}
                    content_path = raw.get("content_path")
                    if not content_path:
                        save_path = raw.get("save_path") or ""
                        name = raw.get("name") or ""
                        content_path = str(Path(save_path) / name) if save_path and name else ""
                    
                    print(f"  [{idx}/{len(to_fetch)}] {truncate(r['name'], 60)}...", end="\r", flush=True)
                    get_mediainfo_summary(r['hash'], content_path)
                print("\nDone.")

        os.system("clear")
        print(f"{COLOR_DEFAULT}{COLOR_BOLD}QBITTORRENT DASHBOARD (TUI) v{VERSION}{COLOR_RESET}")
        print(f"API: {api_url}")
        print(f"Summary: {summary(torrents)}")
        print("")

        scope_label = scope.upper()
        mode_label = {"i": "INFO", "p": "PAUSE/RESUME", "d": "DELETE", "c": "CATEGORY", "t": "TAGS", "v": "VERIFY", "A": "ADD PUBLIC TRACKERS", "Q": "QC TAG MEDIA", "l": "LIST FILES", "m": "MEDIAINFO"}[mode]
        page_label = f"Page {page + 1}/{total_pages}"
        sort_label = f"{sort_field} ({'desc' if sort_desc else 'asc'})"
        mode_col = mode_color(mode)
        print(f"Mode: {mode_col}{mode_label}{COLOR_RESET}  Scope: {COLOR_BRIGHT_CYAN}{scope_label}{COLOR_RESET}  Sort: {COLOR_BRIGHT_PURPLE}{sort_label}{COLOR_RESET}  {page_label}")

        print("")
        name_width = 52
        size_width = 12
        cat_width = 14
        hash_width = 40 if show_full_hash else 6
        hash_label = "Hash"
        added_part = f"{'Added':<16} " if show_added else ""
        header_line = (
            f"{'No':<3} {'ST':<2} "
            f"{'Name':<{name_width}} "
            f"{'Prog':<6} {'Size':<{size_width}} {'DL':<8} {'UL':<8} {'ETA':<6} "
            f"{added_part}{'Cat':<{cat_width}} {hash_label:<{hash_width}}"
        )
        divider_line = "-" * min(len(header_line), terminal_width())
        print(header_line)
        print(divider_line)

        for idx, item in enumerate(page_rows, 0):
            status_col = status_color(item.get("state") or "")
            name = (item.get("name") or "")[:name_width]
            state = item.get("state") or ""
            st = item.get("st") or "?"
            cat_val = str(item.get("category") or "-")
            added_value = f"{str(item.get('added') or '-'): <16} " if show_added else ""
            base_line = (
                f"{idx:<3} {status_col}{st:<2}{COLOR_RESET} "
                f"{status_col}{name:<{name_width}}{COLOR_RESET} "
                f"{str(item.get('progress') or '-'): <6} "
                f"{str(item.get('size') or '-'): <{size_width}} "
                f"{str(item.get('dlspeed') or '-'): <8} "
                f"{str(item.get('upspeed') or '-'): <8} "
                f"{str(item.get('eta') or '-'): <6} "
                f"{added_value}"
                f"{COLOR_BROWN}{cat_val:<{cat_width}}{COLOR_RESET} "
            )
            hash_value = str(item.get("hash") or "")
            hash_display = hash_value if show_full_hash else hash_value[:6] or "-"
            print(f"{base_line}{hash_display:<{hash_width}}")

            if show_mediainfo_inline:
                raw = item.get("raw") or {}
                content_path = raw.get("content_path")
                if not content_path:
                    save_path = raw.get("save_path") or ""
                    name = raw.get("name") or ""
                    content_path = str(Path(save_path) / name) if save_path and name else ""
                
                mi_summary = get_mediainfo_summary(item.get("hash"), content_path)
                indent = "     "
                print(f"{indent}{COLOR_GREY}{mi_summary}{COLOR_RESET}")

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
                    indent = "     tags: "
                    width = max(40, terminal_width() - len(indent))
                    for line in wrap_ansi(tags_line, width):
                        print(indent + line)

        print(divider_line)
        print(format_filters_line(filters))
        print(divider_line)
        print(
            "Keys: r=Refresh f=Filter a=All w=Down u=Up z=Paused e=Done g=Err s=Sort [i/p/d/c/t/v/l/m]=Modes"
        )
        print(
            "      V=View F=Filters P=Presets S=Dir D=Added H=Hash T=Tags C=Cat M=MediaInfo [A/Q]=Modes R=Raw"
        )
        print(
            "      0-9=Apply [=Prev ]=Next #=Tag /=Line ?=Help Ctrl-Q=Quit"
        )
        print(divider_line)

        key = get_key()
        if key == "\x11":
            break
        if key == "?":
            print("Modes: i=info, p=pause/resume, d=delete, c=category, t=tags, v=verify, A=add public trackers (non-private), Q=qc-tag-media, l=list files, m=mediainfo")
            print("Paging: ] next page, [ previous page")
            print("Scope: a=all, w=downloading, u=uploading, z=paused, e=completed, g=error  Raw: R + item number")
            print("Sort: s=cycle field, S=toggle asc/desc  Columns: T=toggle tags, D=toggle added, H=toggle hash width")
            print("Filters: f=text, C=category, #=tag (comma=OR, plus=AND, !NOT, ()group), /=line, F=manage stack, P=presets")
            print("Filter examples: text=anime cat=tv tag=ab+cross | text=!silo cat=- tag=(ab,cross)+!z")
            print("MediaInfo: m=mode, M=toggle inline summary")
            print("Quit: Ctrl-Q")
            print("Press any key to continue...", end="", flush=True)
            _ = get_key()
            continue
        if key == "M":
            show_mediainfo_inline = not show_mediainfo_inline
            continue
        if key == "V":
            mode = "i"
            scope = "all"
            page = 0
            filters = []
            sort_index = 0
            sort_desc = True
            show_tags = False
            show_added = True
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
                negate = value.startswith("!")
                if negate:
                    value = value[1:]
                filters.append({"type": "text", "value": value, "enabled": True, "negate": negate})
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
                negate = value.startswith("!")
                if negate:
                    value = value[1:]
                filters.append({"type": "category", "value": value, "enabled": True, "negate": negate})
            page = 0
            continue
        if key == "#":
            value = read_line("\nTag filter (comma=OR, plus=AND, !NOT, ()group; blank clears): ").strip()
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
                    prefix = "!" if flt.get("negate") else ""
                    label = f"text={prefix}{flt['value']}"
                elif flt["type"] == "category":
                    prefix = "!" if flt.get("negate") else ""
                    label = f"cat={prefix}{flt['value']}"
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
        if key == "D":
            show_added = not show_added
            continue
        if key == "H":
            show_full_hash = not show_full_hash
            continue
        if key in "ipdctvAQlm":
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
                if 0 <= idx < len(page_rows):
                    print("")
                    print_raw(page_rows[idx])
            continue
        if key and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(page_rows):
                item = page_rows[idx]
                if mode == "i":
                    print("")
                    print_details(item)
                elif mode == "l":
                    print("")
                    print_files(opener, api_url, item)
                elif mode == "m":
                    print("")
                    print_mediainfo(item)
                else:
                    result = apply_action(opener, api_url, mode, item)
                    print(f"{mode_label}: {result}")
                    time.sleep(0.6)
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
