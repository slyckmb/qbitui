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
VERSION = "1.5.0"
LAST_UPDATED = "2026-02-02"

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

def color_hex(value: str) -> str:
    value = value.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return f"\033[38;2;{r};{g};{b}m"


COLOR_BG = "\033[48;2;48;10;36m"
COLOR_FG = color_hex("EEEEEC")
COLOR_MUTED = color_hex("D3D7CF")
COLOR_DIVIDER = color_hex("555753")
COLOR_FOCUS = color_hex("729FCF")
COLOR_TAB_ACTIVE = color_hex("FCE94F")
COLOR_TAB_INACTIVE = color_hex("D3D7CF")
COLOR_ACTION_DANGER = color_hex("EF2929")
COLOR_ACTION_CONFIRM = color_hex("FCE94F")
COLOR_STATE_DOWNLOADING = color_hex("8AE234")
COLOR_STATE_UPLOADING = color_hex("34E2E2")
COLOR_STATE_PAUSED = color_hex("FCE94F")
COLOR_STATE_ERROR = color_hex("EF2929")
COLOR_STATE_COMPLETED = color_hex("AD7FA8")
COLOR_STATE_CHECKING = color_hex("729FCF")
def color_bg_hex(value: str) -> str:
    value = value.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return f"\033[48;2;{r};{g};{b}m"

COLOR_SELECTION_FG = color_hex("FFFFFF")
COLOR_SELECTION_BG = color_bg_hex("3465A4")
COLOR_SELECTION = COLOR_SELECTION_BG + COLOR_SELECTION_FG
COLOR_UNDERLINE = "\033[4m"
COLOR_DIM = "\033[2m"
COLOR_DEFAULT = COLOR_FG
COLOR_RESET = "\033[0m" + COLOR_FG

LOCAL_TZ = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
PRESET_FILE = Path(__file__).parent.parent / "config" / "qbit-filter-presets.yml"
TRACKERS_LIST_URL = "https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_best.txt"
QC_TAG_TOOL = Path(__file__).resolve().parent / "media_qc_tag.py"
QC_LOG_DIR = Path.home() / ".logs" / "media_qc"

MEDIA_EXTS = {
    # Video
    ".3g2", ".3gp", ".asf", ".asx", ".avi", ".divx", ".f4v", ".flv", ".m2p", ".m2ts",
    ".m2v", ".m4v", ".mjp", ".mkv", ".mov", ".mp4", ".mpe", ".mpeg", ".mpg", ".mts",
    ".ogm", ".ogv", ".qt", ".rm", ".rmvb", ".swf", ".ts", ".vob", ".webm", ".wmv",
    ".xvid",
    # Audio
    ".aa3", ".aac", ".ac3", ".acm", ".adts", ".aif", ".aifc", ".aiff", ".amr", ".ape",
    ".au", ".caf", ".dts", ".flac", ".fla", ".m4a", ".m4b", ".m4p", ".mid", ".mka",
    ".mod", ".mp2", ".mp3", ".mp4", ".mpc", ".oga", ".ogg", ".opus", ".ra", ".ram",
    ".wav", ".wma", ".wv"
}

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


STASHED_KEY = ""


def get_key() -> str:
    """Get single keypress."""
    global STASHED_KEY
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if STASHED_KEY:
            ch = STASHED_KEY
            STASHED_KEY = ""
            return ch
        def read_with_timeout(timeout: float) -> str:
            start = time.monotonic()
            while time.monotonic() - start < timeout:
                if select.select([sys.stdin], [], [], 0.02)[0]:
                    return sys.stdin.read(1)
            return ""

        ch = sys.stdin.read(1)
        if ch == "\x1b":
            nxt = read_with_timeout(0.3)
            if not nxt:
                return "ESC"
            if nxt == "[":
                seq = ""
                while True:
                    part = read_with_timeout(0.15)
                    if not part:
                        break
                    seq += part
                    if seq.endswith(("A", "B", "C", "D", "I", "Z")):
                        break
                    if len(seq) >= 6:
                        break
                if seq and seq[-1] in "ABCD":
                    return ""
                if seq.startswith("1;5") and seq.endswith(("I", "Z")):
                    return "CTRL_TAB"
                if seq.startswith("1;6") and seq.endswith(("I", "Z")):
                    return "CTRL_TAB"
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
        return COLOR_STATE_ERROR
    if s in {"downloading", "forcedDL", "metaDL"}:
        return COLOR_STATE_DOWNLOADING
    if s in {"uploading", "forcedUP"}:
        return COLOR_STATE_UPLOADING
    if s in {"pausedDL", "pausedUP"}:
        return COLOR_STATE_PAUSED
    if s in {"completed"}:
        return COLOR_STATE_COMPLETED
    if s in {"queuedDL", "queuedUP", "checkingUP", "checkingDL", "checkingResumeData", "queuedForChecking", "checking"}:
        return COLOR_STATE_CHECKING
    if s in {"stalledUP", "stalledDL", "allocating", "moving"}:
        return COLOR_ORANGE
    if not s:
        return COLOR_FG
    return COLOR_FG


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
    
    files = []
    if path.is_file():
        if path.suffix.lower() in MEDIA_EXTS:
            files.append(path)
    else:
        for item_path in path.rglob("*"):
            if item_path.is_file() and item_path.suffix.lower() in MEDIA_EXTS:
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
        val = cache_file.read_text().strip()
        # Invalidate old error messages or formats without dots
        if val and " • " in val and not val.startswith("MediaInfo"):
            return val

    target = get_largest_media_file(content_path)
    if not target:
        mi_summary = "No media content."
        cache_file.write_text(mi_summary)
        return mi_summary

    tool = shutil.which("mediainfo")
    if not tool:
        return "ERROR: mediainfo not found"

    # Template covers General, Video, and Audio tracks.
    inform = "General;%Format%|%Duration/String3%|%OverallBitRate/String%|Video;%Width%x%Height% %Format% %BitRate/String%|Audio;%Format% %Channel(s)%ch"
    
    result = subprocess.run(
        [tool, f"--Inform={inform}", str(target)],
        capture_output=True,
        text=True,
    )
    
    res = (result.stdout or "").strip()
    if not res:
        # Fallback for files where track-specific templates might fail
        result = subprocess.run([tool, "--Inform=General;%Format% %Duration/String3%", str(target)], capture_output=True, text=True)
        res = (result.stdout or "").strip()

    if not res:
        mi_summary = "MediaInfo extraction failed."
    else:
        # Split by pipes and join with dots for a clean 'info line' look
        parts = [p.strip() for p in res.split("|") if p.strip()]
        mi_summary = " • ".join(parts)
    
    # Final cleanup
    mi_summary = mi_summary.replace("  ", " ").strip()
    
    cache_file.write_text(mi_summary)
    return mi_summary


def get_mediainfo_summary_cached(hash_value: str, content_path: str) -> str:
    cache_file = CACHE_DIR / f"{hash_value}.summary"
    if cache_file.exists():
        return cache_file.read_text().strip()
    return get_mediainfo_summary(hash_value, content_path)


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
        for item_path in sorted(path.rglob("*")):
            if item_path.is_file() and item_path.suffix.lower() in MEDIA_EXTS:
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


def confirm_delete(item: dict) -> tuple[bool, bool]:
    name = item.get("name") or "Unknown"
    hash_value = item.get("hash") or "unknown"
    remove_ok = read_line("Remove torrent? (y/N): ").strip().lower() == "y"
    if not remove_ok:
        return False, False
    delete_files = read_line("Delete data too? (y/N): ").strip().lower() == "y"
    summary = f"Confirm remove {'+ delete data ' if delete_files else ''}{name} ({hash_value})? (y/N): "
    final_ok = read_line(summary).strip().lower() == "y"
    if not final_ok:
        return False, False
    return True, delete_files


def apply_action(opener: urllib.request.OpenerDirector, api_url: str, action: str, item: dict) -> str:
    hash_value = item.get("hash")
    if not hash_value:
        return "Missing hash"
    state = item.get("state") or ""
    raw = item.get("raw", {})

    if action == "P":
        is_paused = "paused" in state.lower() or "stopped" in state.lower()
        action = "start" if is_paused else "stop"
        resp = qbit_request(opener, api_url, "POST", f"/api/v2/torrents/{action}", {"hashes": hash_value})
        # Try fallbacks for older versions if start/stop 404
        if "HTTP 404" in resp:
            old_action = "resume" if is_paused else "pause"
            resp = qbit_request(opener, api_url, "POST", f"/api/v2/torrents/{old_action}", {"hashes": hash_value})
        return "OK" if resp in ("Ok.", "") else resp
    if action == "D":
        confirmed, delete_files = confirm_delete(item)
        if not confirmed:
            return "Cancelled"
        resp = qbit_request(
            opener,
            api_url,
            "POST",
            "/api/v2/torrents/delete",
            {"hashes": hash_value, "deleteFiles": "true" if delete_files else "false"},
        )
        return "OK" if resp in ("Ok.", "") else resp
    if action == "C":
        value = read_line("Enter new category (blank cancels): ").strip()
        if not value:
            return "Cancelled"
        resp = qbit_request(opener, api_url, "POST", "/api/v2/torrents/setCategory", {"hashes": hash_value, "category": value})
        return "OK" if resp in ("Ok.", "") else resp
    if action == "E":
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
    if action == "V":
        resp = qbit_request(opener, api_url, "POST", "/api/v2/torrents/recheck", {"hashes": hash_value})
        return "OK" if resp in ("Ok.", "") else resp
    if action == "A":
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
    if action == "Q":
        return spawn_media_qc(hash_value)
    return "Unknown action"


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


def fetch_trackers(opener: urllib.request.OpenerDirector, api_url: str, hash_value: str) -> list[dict]:
    raw = qbit_request(opener, api_url, "GET", "/api/v2/torrents/trackers", {"hash": hash_value})
    try:
        return json.loads(raw) if raw else []
    except Exception:
        return []


def fetch_files(opener: urllib.request.OpenerDirector, api_url: str, hash_value: str) -> list[dict]:
    raw = qbit_request(opener, api_url, "GET", "/api/v2/torrents/files", {"hash": hash_value})
    try:
        return json.loads(raw) if raw else []
    except Exception:
        return []


def fetch_peers(opener: urllib.request.OpenerDirector, api_url: str, hash_value: str) -> dict:
    raw = qbit_request(opener, api_url, "GET", "/api/v2/sync/torrentPeers", {"hash": hash_value})
    try:
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def render_info_lines(item: dict, width: int) -> list[str]:
    raw = item.get("raw") or {}
    lines = [
        f"Name: {item.get('name')}",
        f"State: {item.get('state')}",
        f"Category: {item.get('category')}",
        f"Tags: {item.get('tags')}",
        f"Size: {item.get('size')}",
        f"Progress: {item.get('progress')}",
        f"Ratio: {item.get('ratio')}",
        f"DL/UL: {item.get('dlspeed')} / {item.get('upspeed')}",
        f"ETA: {item.get('eta')}",
        f"Hash: {item.get('hash')}",
    ]
    for key in ("save_path", "content_path", "tracker", "completion_on", "added_on", "last_activity"):
        if key in raw:
            value = raw.get(key)
            if key.endswith("_on") and isinstance(value, (int, float)):
                value = format_ts(value)
            lines.append(f"{key}: {value}")
    wrapped = []
    for line in lines:
        wrapped.extend(wrap_ansi(line, width))
    return wrapped


def render_trackers_lines(trackers: list[dict], width: int, max_rows: int) -> list[str]:
    if not trackers:
        return ["No trackers."]
    headers = ["Status", "Tier", "URL"]
    widths = [10, 6, max(20, width - 20)]
    lines = []
    lines.append(f"{headers[0]:<{widths[0]}} {headers[1]:<{widths[1]}} {headers[2]}")
    lines.append("-" * min(width, widths[0] + widths[1] + widths[2] + 2))
    for row in trackers[:max_rows]:
        status = str(row.get("status", ""))
        tier = str(row.get("tier", ""))
        url = str(row.get("url", ""))
        url = truncate(url, widths[2])
        lines.append(f"{status:<{widths[0]}} {tier:<{widths[1]}} {url}")
    if len(trackers) > max_rows:
        lines.append(f"... ({len(trackers) - max_rows} more)")
    return lines


def render_files_lines(files: list[dict], width: int, max_rows: int) -> list[str]:
    if not files:
        return ["No files found."]
    files.sort(key=lambda x: x.get("name", ""))
    headers = ["Index", "Name", "Size", "Prog", "Priority"]
    widths = [5, max(20, width - 32), 10, 6, 10]
    lines = []
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * min(width, len(header_line)))
    priority_map = {0: "Do not DL", 1: "Normal", 2: "High", 6: "Max", 7: "Forced"}
    for idx, f in enumerate(files[:max_rows]):
        name = truncate(f.get("name", ""), widths[1])
        size = size_str(f.get("size", 0))
        prog = f"{int(f.get('progress', 0) * 100)}%"
        prio = priority_map.get(f.get("priority", 1), str(f.get("priority")))
        line = (
            f"{str(idx):<5} "
            f"{name:<{widths[1]}} "
            f"{size:<10} "
            f"{prog:<6} "
            f"{prio:<10}"
        )
        lines.append(line)
    if len(files) > max_rows:
        lines.append(f"... ({len(files) - max_rows} more)")
    return lines


def render_peers_lines(peers_payload: dict, width: int, max_rows: int) -> list[str]:
    peers = peers_payload.get("peers") or {}
    if not peers:
        return ["No peers."]
    rows = []
    for addr, info in peers.items():
        dl_speed = info.get("dl_speed") or 0
        ul_speed = info.get("up_speed") or 0
        rows.append({
            "addr": addr,
            "client": info.get("client", ""),
            "progress": int((info.get("progress", 0) or 0) * 100),
            "dl": speed_str(dl_speed),
            "ul": speed_str(ul_speed),
            "dl_raw": dl_speed,
            "ul_raw": ul_speed,
            "flags": info.get("flags", ""),
        })
    rows.sort(key=lambda x: (x["dl_raw"], x["ul_raw"]), reverse=True)
    headers = ["Peer", "Prog", "DL", "UL", "Flags", "Client"]
    widths = [18, 6, 10, 10, 8, max(20, width - 60)]
    lines = []
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("-" * min(width, len(header_line)))
    for row in rows[:max_rows]:
        client = truncate(row["client"], widths[5])
        line = (
            f"{row['addr']:<{widths[0]}} "
            f"{str(row['progress']) + '%':<{widths[1]}} "
            f"{row['dl']:<{widths[2]}} "
            f"{row['ul']:<{widths[3]}} "
            f"{row['flags']:<{widths[4]}} "
            f"{client:<{widths[5]}}"
        )
        lines.append(line)
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more)")
    return lines


def render_mediainfo_lines(item: dict, width: int) -> list[str]:
    raw = item.get("raw") or {}
    content_path = raw.get("content_path")
    if not content_path:
        save_path = raw.get("save_path") or ""
        name = raw.get("name") or ""
        content_path = str(Path(save_path) / name) if save_path and name else ""
    info = get_mediainfo_for_hash(item.get("hash"), content_path)
    lines = []
    for line in str(info).splitlines():
        lines.extend(wrap_ansi(line, width))
    return lines or ["No MediaInfo."]


def resolve_available_tabs(opener: urllib.request.OpenerDirector, api_url: str, item: dict) -> list[str]:
    available = ["Info"]
    hash_value = item.get("hash")
    if not hash_value:
        return available
    trackers = fetch_trackers(opener, api_url, hash_value)
    if trackers:
        available.append("Trackers")
    files = fetch_files(opener, api_url, hash_value)
    if files:
        available.append("Content")
    peers_payload = fetch_peers(opener, api_url, hash_value)
    if peers_payload.get("peers"):
        available.append("Peers")
    raw = item.get("raw") or {}
    content_path = raw.get("content_path")
    if not content_path:
        save_path = raw.get("save_path") or ""
        name = raw.get("name") or ""
        content_path = str(Path(save_path) / name) if save_path and name else ""
    if content_path and shutil.which("mediainfo") and get_largest_media_file(content_path):
        available.append("MediaInfo")
    return available


def capture_key_sequences() -> None:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    print(f"\nDebug key capture v{VERSION}: press Ctrl-Q to exit.", flush=True)
    try:
        tty.setraw(fd)
        esc_pending = False
        esc_buffer = b""
        bracket_pending = False
        bracket_started = 0.0
        while True:
            buf = sys.stdin.buffer.read(1)
            if not buf:
                continue
            if buf == b"\x11":
                print("EXIT", flush=True)
                break
            if bracket_pending and time.monotonic() - bracket_started > 1.0:
                bracket_pending = False
            if buf == b"\x1b":
                esc_pending = True
                esc_buffer = b""
                continue
            if buf == b"[" and not esc_pending:
                bracket_pending = True
                bracket_started = time.monotonic()
                continue
            if bracket_pending:
                bracket_pending = False
                if buf in (b"A", b"B", b"C", b"D"):
                    seq = b"\x5b" + buf
                    hex_bytes = " ".join(f"{b:02x}" for b in seq)
                    print(f"SEQ {hex_bytes}  {seq!r}", flush=True)
                    continue
            if esc_pending:
                if not esc_buffer and buf in (b"[", b"O"):
                    esc_buffer = buf
                    continue
                esc_buffer += buf
                if esc_buffer.endswith((b"A", b"B", b"C", b"D")):
                    seq = b"\x1b" + esc_buffer
                    hex_bytes = " ".join(f"{b:02x}" for b in seq)
                    print(f"SEQ {hex_bytes}  {seq!r}", flush=True)
                    esc_pending = False
                    esc_buffer = b""
                continue
            if esc_pending:
                print("ESC", flush=True)
                esc_pending = False
                esc_buffer = b""
            hex_bytes = " ".join(f"{b:02x}" for b in buf)
            print(f"KEY {hex_bytes}  {buf!r}", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive qBittorrent dashboard")
    parser.add_argument("--config", default=os.environ.get("QBITTORRENT_CONFIG_FILE"), help="Path to request-cache.yml")
    parser.add_argument("--page-size", type=int, default=int(os.environ.get("QBITTORRENT_PAGE_SIZE", "10")))
    parser.add_argument("--debug-keys", help="Write raw key sequences to a file (TTY only).")
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
    focus_idx = 0
    selection_hash: str | None = None
    selection_name: str | None = None
    in_tab_view = False
    tabs = ["Info", "Trackers", "Content", "Peers", "MediaInfo"]
    active_tab = 0
    banner_text = ""
    banner_until = 0.0
    last_banner_time = 0.0

    def set_banner(message: str, duration: float = 2.0, min_interval: float = 0.6) -> None:
        nonlocal banner_text, banner_until, last_banner_time
        now = time.time()
        if banner_text == message and now - last_banner_time < min_interval:
            return
        banner_text = message
        banner_until = now + duration
        last_banner_time = now

    if args.debug_keys:
        try:
            log_path = Path(args.debug_keys).expanduser()
        except Exception:
            log_path = Path(args.debug_keys)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as handle:
            handle.write(f"\n=== key capture v{VERSION} {datetime.now(LOCAL_TZ).isoformat()} ===\n")
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                buf = sys.stdin.buffer.read(1)
                if not buf:
                    continue
                if buf == b"\x11":
                    with log_path.open("a") as handle:
                        handle.write("EXIT (Ctrl-Q)\n")
                    break
                if buf == b"\x1b":
                    start = time.monotonic()
                    rest = b""
                    while time.monotonic() - start < 0.25:
                        if select.select([sys.stdin], [], [], 0.02)[0]:
                            rest += sys.stdin.buffer.read(1)
                            if rest.endswith((b"A", b"B", b"C", b"D")):
                                break
                        else:
                            if rest:
                                break
                    if not rest:
                        with log_path.open("a") as handle:
                            handle.write("ESC\n")
                        continue
                    seq = buf + rest
                    hex_bytes = " ".join(f"{b:02x}" for b in seq)
                    with log_path.open("a") as handle:
                        handle.write(f"SEQ {hex_bytes}  {seq!r}\n")
                    continue
                hex_bytes = " ".join(f"{b:02x}" for b in buf)
                with log_path.open("a") as handle:
                    handle.write(f"KEY {hex_bytes}  {buf!r}\n")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return 0

    cached_torrents: list[dict] = []
    cached_rows: list[dict] = []
    cache_time = 0.0
    fetch_interval = 2.0
    list_start_row = 0
    list_block_height = 0
    have_full_draw = False
    mi_bootstrap_done = False
    mi_queue: list[str] = []
    mi_queue_index = 0
    mi_last_tick = 0.0

    def print_at(row: int, text: str) -> None:
        sys.stdout.write(f"\033[{row};1H\033[2K{text}")

    def build_list_block(page_rows_local: list[dict]) -> list[str]:
        lines: list[str] = []
        lines.append(header_line)
        lines.append(divider_line)
        for idx, item in enumerate(page_rows_local, 0):
            selected = selection_hash == item.get("hash")
            status_col = status_color(item.get("state") or "")
            name = (item.get("name") or "")[:name_width]
            st = item.get("st") or "?"
            cat_val = str(item.get("category") or "-")
            added_value = f"{str(item.get('added') or '-'): <16} " if show_added else ""
            hash_value = str(item.get("hash") or "")
            hash_display = hash_value if show_full_hash else hash_value[:6] or "-"
            focus_marker = ">" if idx == focus_idx else " "
            if selected:
                base_line = (
                    f"{focus_marker:<1} {idx:<2} {st:<2} "
                    f"{name:<{name_width}} "
                    f"{str(item.get('progress') or '-'): <6} "
                    f"{str(item.get('size') or '-'): <{size_width}} "
                    f"{str(item.get('dlspeed') or '-'): <8} "
                    f"{str(item.get('upspeed') or '-'): <8} "
                    f"{str(item.get('eta') or '-'): <6} "
                    f"{added_value}"
                    f"{cat_val:<{cat_width}} "
                    f"{hash_display:<{hash_width}}"
                )
                lines.append(f"{COLOR_SELECTION}{base_line}{COLOR_RESET}")
            else:
                focus_col = f"{COLOR_FOCUS}{focus_marker}{COLOR_RESET}" if idx == focus_idx else " "
                base_line = (
                    f"{focus_col:<1} {idx:<2} {status_col}{st:<2}{COLOR_RESET} "
                    f"{status_col}{name:<{name_width}}{COLOR_RESET} "
                    f"{str(item.get('progress') or '-'): <6} "
                    f"{str(item.get('size') or '-'): <{size_width}} "
                    f"{str(item.get('dlspeed') or '-'): <8} "
                    f"{str(item.get('upspeed') or '-'): <8} "
                    f"{str(item.get('eta') or '-'): <6} "
                    f"{added_value}"
                    f"{COLOR_BROWN}{cat_val:<{cat_width}}{COLOR_RESET} "
                    f"{hash_display:<{hash_width}}"
                )
                lines.append(base_line)

            if show_mediainfo_inline:
                raw_item = item.get("raw") or {}
                content_path = raw_item.get("content_path")
                if not content_path:
                    save_path = raw_item.get("save_path") or ""
                    item_name = raw_item.get("name") or ""
                    content_path = str(Path(save_path) / item_name) if save_path and item_name else ""
                mi_summary = get_mediainfo_summary_cached(item.get("hash"), content_path)
                indent = "     "
                lines.append(f"{indent}{COLOR_GREY}{mi_summary}{COLOR_RESET}")

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
                    width = max(40, term_w - len(indent))
                    for line in wrap_ansi(tags_line, width):
                        lines.append(indent + line)
        return lines

    def update_mediainfo_cache(rows_sorted: list[dict], total_pages_local: int) -> None:
        nonlocal mi_bootstrap_done, mi_queue, mi_queue_index, mi_last_tick
        if not rows_sorted:
            return
        now_tick = time.monotonic()
        if now_tick - mi_last_tick < 0.2:
            return
        if not mi_bootstrap_done:
            targets = []
            for idx in {0, 1, max(0, total_pages_local - 2)}:
                start = idx * args.page_size
                end = min(start + args.page_size, len(rows_sorted))
                targets.extend(rows_sorted[start:end])
            for item in targets:
                hash_value = item.get("hash") or ""
                if not hash_value:
                    continue
                cache_file = CACHE_DIR / f"{hash_value}.summary"
                if not cache_file.exists() and hash_value not in mi_queue:
                    mi_queue.append(hash_value)
            mi_bootstrap_done = True
        if not mi_queue:
            for item in rows_sorted:
                hash_value = item.get("hash") or ""
                if not hash_value:
                    continue
                cache_file = CACHE_DIR / f"{hash_value}.summary"
                if not cache_file.exists():
                    mi_queue.append(hash_value)
            mi_queue_index = 0
        if mi_queue:
            hash_value = mi_queue[mi_queue_index % len(mi_queue)]
            mi_queue_index += 1
            item = next((r for r in rows_sorted if r.get("hash") == hash_value), None)
            if item:
                raw_item = item.get("raw") or {}
                content_path = raw_item.get("content_path")
                if not content_path:
                    save_path = raw_item.get("save_path") or ""
                    item_name = raw_item.get("name") or ""
                    content_path = str(Path(save_path) / item_name) if save_path and item_name else ""
                get_mediainfo_summary_cached(hash_value, content_path)
        mi_last_tick = now_tick

    while True:
        now = time.monotonic()
        if cached_rows and (now - cache_time) < fetch_interval:
            torrents = cached_torrents
            rows = cached_rows
        else:
            raw = qbit_request(opener, api_url, "GET", "/api/v2/torrents/info")
            try:
                torrents = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                torrents = []
            rows = build_rows(torrents)
            cached_torrents = torrents
            cached_rows = rows
            cache_time = now

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

        if focus_idx >= len(page_rows):
            focus_idx = max(0, len(page_rows) - 1)

        selected_row_all = None
        if selection_hash:
            for row in rows:
                if row.get("hash") == selection_hash:
                    selected_row_all = row
                    break
            if not selected_row_all:
                selection_hash = None
                selection_name = None
                in_tab_view = False
                active_tab = 0
                set_banner("Selection cleared: item removed.")
            else:
                if not any(r.get("hash") == selection_hash for r in page_rows):
                    selection_hash = None
                    selection_name = None
                    in_tab_view = False
                    active_tab = 0
                    set_banner("Selection cleared: moved off page.")
                else:
                    selection_name = selected_row_all.get("name") or selection_name

        # Pre-populate MediaInfo cache if toggle is on
        update_mediainfo_cache(rows, total_pages)

        term_w = terminal_width()
        banner_line = ""
        if banner_text and time.time() < banner_until:
            banner_line = f"{COLOR_ACTION_CONFIRM}{banner_text}{COLOR_RESET}"
        elif not selection_hash:
            banner_line = f"{COLOR_MUTED}Select an item.{COLOR_RESET}"
        else:
            short_hash = (selection_hash or "")[:10]
            banner_line = f"{COLOR_TAB_ACTIVE}Selected:{COLOR_RESET} {selection_name} ({short_hash})"

        scope_label = scope.upper()
        page_label = f"Page {page + 1}/{total_pages}"
        sort_label = f"{sort_field} ({'desc' if sort_desc else 'asc'})"

        name_width = 52
        size_width = 12
        cat_width = 14
        hash_width = 40 if show_full_hash else 6
        hash_label = "Hash"
        added_part = f"{'Added':<16} " if show_added else ""

        header_line = (
            f"{'F':<1} {'No':<2} {'ST':<2} "
            f"{'Name':<{name_width}} "
            f"{'Prog':<6} {'Size':<{size_width}} {'DL':<8} {'UL':<8} {'ETA':<6} "
            f"{added_part}{'Cat':<{cat_width}} {hash_label:<{hash_width}}"
        )
        divider_line = "-" * min(len(header_line), term_w)

        list_block_lines = build_list_block(page_rows)

        if in_tab_view and selection_hash:
            os.system("clear")
            print(f"{COLOR_DEFAULT}{COLOR_BOLD}QBITTORRENT DASHBOARD (TUI) v{VERSION}{COLOR_RESET}")
            print(f"API: {api_url}")
            print(f"Summary: {summary(torrents)}")
            print(banner_line)
            print("")
            print(f"Scope: {COLOR_FOCUS}{scope_label}{COLOR_RESET}  Sort: {COLOR_BRIGHT_PURPLE}{sort_label}{COLOR_RESET}  {page_label}")
            print("")
            selected_row = next((r for r in page_rows if r.get("hash") == selection_hash), None)
            if not selected_row:
                print("Selection not available on this page.")
                print(divider_line)
            else:
                available_tabs = resolve_available_tabs(opener, api_url, selected_row)
                if not available_tabs:
                    available_tabs = ["Info"]
                active_label = tabs[active_tab]
                if active_label not in available_tabs:
                    active_tab = tabs.index(available_tabs[0])
                    active_label = tabs[active_tab]
                tab_labels = []
                for label in available_tabs:
                    if label == active_label:
                        tab_labels.append(f"{COLOR_TAB_ACTIVE}[{label}]{COLOR_RESET}")
                    else:
                        tab_labels.append(f"{COLOR_TAB_INACTIVE}{label}{COLOR_RESET}")
                print("Tabs: " + " ".join(tab_labels))
                print(divider_line)
                tab_width = max(40, term_w - 4)
                max_rows = max(10, shutil.get_terminal_size((100, 30)).lines - 14)
                if active_label == "Info":
                    content_lines = render_info_lines(selected_row, tab_width)
                elif active_label == "Trackers":
                    trackers = fetch_trackers(opener, api_url, selection_hash)
                    content_lines = render_trackers_lines(trackers, tab_width, max_rows)
                elif active_label == "Content":
                    files = fetch_files(opener, api_url, selection_hash)
                    content_lines = render_files_lines(files, tab_width, max_rows)
                elif active_label == "Peers":
                    peers_payload = fetch_peers(opener, api_url, selection_hash)
                    content_lines = render_peers_lines(peers_payload, tab_width, max_rows)
                else:
                    content_lines = render_mediainfo_lines(selected_row, tab_width)
                for line in content_lines[:max_rows]:
                    print(line)
                print(divider_line)
        else:
            if not have_full_draw:
                os.system("clear")
                print(f"{COLOR_DEFAULT}{COLOR_BOLD}QBITTORRENT DASHBOARD (TUI) v{VERSION}{COLOR_RESET}")
                print(f"API: {api_url}")
                print(f"Summary: {summary(torrents)}")
                print(banner_line)
                print("")
                print(f"Scope: {COLOR_FOCUS}{scope_label}{COLOR_RESET}  Sort: {COLOR_BRIGHT_PURPLE}{sort_label}{COLOR_RESET}  {page_label}")
                print("")
                list_start_row = 8
                for line in list_block_lines:
                    print(line)
                print(divider_line)
                list_block_height = len(list_block_lines) + 1
                have_full_draw = True
            else:
                print_at(3, f"Summary: {summary(torrents)}")
                print_at(4, banner_line)
                print_at(6, f"Scope: {COLOR_FOCUS}{scope_label}{COLOR_RESET}  Sort: {COLOR_BRIGHT_PURPLE}{sort_label}{COLOR_RESET}  {page_label}")
                row = list_start_row
                for _ in range(list_block_height):
                    print_at(row, "")
                    row += 1
                row = list_start_row
                for line in list_block_lines:
                    print_at(row, line)
                    row += 1
                print_at(row, divider_line)
                list_block_height = len(list_block_lines) + 1

        print(format_filters_line(filters))
        print(divider_line)

        list_active = f"{COLOR_TAB_ACTIVE}List{COLOR_RESET}" if not in_tab_view else f"{COLOR_MUTED}List{COLOR_RESET}"
        tabs_active = f"{COLOR_TAB_ACTIVE}Tabs{COLOR_RESET}" if in_tab_view else f"{COLOR_MUTED}Tabs{COLOR_RESET}"
        actions_label = "Actions"
        actions_line = f"P=Pause/Resume V=Verify C=Category E=Tags A=Trackers Q=QC {COLOR_ACTION_DANGER}D=Delete{COLOR_RESET}"
        if not selection_hash:
            actions_line = f"{COLOR_DIM}P=Pause/Resume V=Verify C=Category E=Tags A=Trackers Q=QC D=Delete{COLOR_RESET}"
        print(f"{list_active}: a=all w=down u=up v=paused e=done g=err  s=sort o=order  f=text c=cat #=tag l=line  x=filters p=presets  z=reset")
        if selection_hash or in_tab_view:
            print(f"{tabs_active}: Tab=cycle (off after last)  Ctrl-Tab=cycle  T=cycle  Esc=back")
        print(f"{actions_label}: {actions_line}")
        print(f"View: t=tags d=added n=hash m=mediainfo  Nav: ' up  / down  , prev  . next  Space/Enter selects/clears  0-9 selects  `=debug")
        print(divider_line)

        key = get_key()
        if key == "\x11":
            break
        if key == "X":
            if CACHE_DIR.exists():
                shutil.rmtree(CACHE_DIR)
                set_banner("MediaInfo cache cleared.")
            continue
        if key == "?":
            print("Navigation: '(up) /(down) focus, , prev / . next page, Space select/clear, 0-9 select item")
            print("Scope: a=all, w=downloading, u=uploading, v=paused, e=completed, g=error")
            print("Sort: s=cycle field, o=toggle asc/desc")
            print("Filters: f=text, c=category, #=tag, l=line, x=manage stack, p=presets")
            print("Tabs: Tab=cycle tabs (off after last), Ctrl-Tab=cycle, T=cycle (selection required)")
            print("View: t=tags, d=added, n=hash width, m=inline mediainfo, X=clear mediainfo cache")
            print("Reset: z=default view (page 1, newest first)")
            print("Actions (selection required): P pause/resume, V verify, C category, E tags, A add trackers, Q qc, D delete")
            print("Esc clears selection (list) or exits tabs. Quit: Ctrl-Q")
            print("Debug: ` shows raw key sequences (Ctrl-Q to exit)")
            print("Press any key to continue...", end="", flush=True)
            _ = get_key()
            continue
        if key == "`":
            capture_key_sequences()
            continue
        if key == "ESC":
            if in_tab_view:
                in_tab_view = False
                continue
            if selection_hash:
                selection_hash = None
                selection_name = None
                set_banner("Selection cleared.")
            continue
        if key == "CTRL_TAB":
            if selection_hash:
                if not in_tab_view:
                    in_tab_view = True
                selected_row = next((r for r in page_rows if r.get("hash") == selection_hash), None)
                if selected_row:
                    available = resolve_available_tabs(opener, api_url, selected_row)
                    if not available:
                        available = ["Info"]
                    current_label = tabs[active_tab]
                    if current_label not in available:
                        active_tab = tabs.index(available[0])
                    else:
                        idx = available.index(current_label)
                        next_label = available[(idx + 1) % len(available)]
                        active_tab = tabs.index(next_label)
            continue
        if key == "'":
            if page_rows:
                focus_idx = max(0, focus_idx - 1)
            continue
        if key == "/":
            if page_rows:
                focus_idx = min(len(page_rows) - 1, focus_idx + 1)
            continue
        if key == " ":
            if page_rows and 0 <= focus_idx < len(page_rows):
                focused = page_rows[focus_idx]
                if selection_hash and focused.get("hash") == selection_hash:
                    selection_hash = None
                    selection_name = None
                    set_banner("Selection cleared.")
                else:
                    selection_hash = focused.get("hash")
                    selection_name = focused.get("name")
            continue
        if key in ("\r", "\n"):
            if page_rows and 0 <= focus_idx < len(page_rows):
                focused = page_rows[focus_idx]
                if selection_hash and focused.get("hash") == selection_hash:
                    selection_hash = None
                    selection_name = None
                    set_banner("Selection cleared.")
                else:
                    selection_hash = focused.get("hash")
                    selection_name = focused.get("name")
            continue
        if key == "\t":
            if not selection_hash:
                set_banner("Select an item.")
                continue
            selected_row = next((r for r in page_rows if r.get("hash") == selection_hash), None)
            if not selected_row:
                set_banner("Selection cleared: moved off page.")
                selection_hash = None
                selection_name = None
                in_tab_view = False
                continue
            available = resolve_available_tabs(opener, api_url, selected_row)
            if not available:
                available = ["Info"]
            if not in_tab_view:
                in_tab_view = True
                if tabs[active_tab] not in available:
                    active_tab = tabs.index(available[0])
                continue
            current_label = tabs[active_tab]
            if current_label not in available:
                active_tab = tabs.index(available[0])
                continue
            idx = available.index(current_label)
            if idx == len(available) - 1:
                in_tab_view = False
            else:
                next_label = available[idx + 1]
                active_tab = tabs.index(next_label)
            continue
        if key == "T":
            if in_tab_view and selection_hash:
                selected_row = next((r for r in page_rows if r.get("hash") == selection_hash), None)
                if selected_row:
                    available = resolve_available_tabs(opener, api_url, selected_row)
                    if not available:
                        available = ["Info"]
                    current_label = tabs[active_tab]
                    if current_label not in available:
                        active_tab = tabs.index(available[0])
                    else:
                        idx = available.index(current_label)
                        next_label = available[(idx + 1) % len(available)]
                        active_tab = tabs.index(next_label)
            continue
        if key == "m":
            show_mediainfo_inline = not show_mediainfo_inline
            continue
        if key == "z":
            scope = "all"
            page = 0
            sort_index = 0
            sort_desc = True
            filters = []
            show_tags = False
            show_mediainfo_inline = False
            show_full_hash = False
            show_added = True
            focus_idx = 0
            in_tab_view = False
            active_tab = 0
            selection_hash = None
            selection_name = None
            set_banner("View reset.")
            continue
        if key == "r":
            cache_time = 0.0
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
        if key == "v":
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
        if key == "l":
            line = read_line("\nFilter line (text=... cat=... tag=...): ").strip()
            if line:
                filters = parse_filter_line(line, filters)
                page = 0
            continue
        if key == "c":
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
        if key == "p":
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
        if key == "x":
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
        if key == "o":
            sort_desc = not sort_desc
            page = 0
            continue
        if key == "t":
            show_tags = not show_tags
            continue
        if key == "d":
            show_added = not show_added
            continue
        if key == "n":
            show_full_hash = not show_full_hash
            continue
        if key == ",":
            page = total_pages - 1 if page == 0 else page - 1
            focus_idx = 0
            continue
        if key == ".":
            page = 0 if page >= total_pages - 1 else page + 1
            focus_idx = 0
            continue
        if key and key.isdigit():
            idx = int(key)
            if 0 <= idx < len(page_rows):
                selection_hash = page_rows[idx].get("hash")
                selection_name = page_rows[idx].get("name")
                focus_idx = idx
            continue
        if key and key.isupper():
            if not selection_hash:
                continue
            if in_tab_view:
                continue
            selected_row = next((r for r in page_rows if r.get("hash") == selection_hash), None)
            if not selected_row:
                set_banner("Selection cleared: moved off page.")
                selection_hash = None
                selection_name = None
                continue
            if key in {"P", "V", "C", "E", "A", "Q", "D"}:
                result = apply_action(opener, api_url, key, selected_row)
                set_banner(f"{key}: {result}")
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
