#!/usr/bin/env python3
"""Interactive qBittorrent dashboard with modes, hotkeys, and paging."""
import argparse
import json
import os
import shlex
import select
import shutil
import sys
import signal
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
VERSION = "1.7.3"
LAST_UPDATED = "2026-02-06"

# ============================================================================
# COLOR SYSTEM - Claude Code Dark Mode Inspired
# ============================================================================

class ColorScheme:
    """Color scheme management with YAML override support."""

    def __init__(self, yaml_path: Optional[Path] = None):
        """Initialize colors from YAML file or defaults."""
        self.BOLD = "\033[1m"
        self.DIM = "\033[2m"
        self.UNDERLINE = "\033[4m"
        self.RESET = "\033[0m"

        # Default palette (Claude Code Dark Mode inspired)
        self._palette = {
            'bg_primary': '#2C001E',
            'bg_secondary': '#352840',
            'bg_selected': '#3A2F5F',
            'fg_primary': '#E5E7EB',
            'fg_secondary': '#A0AEC0',
            'fg_tertiary': '#6B7280',
            'cyan': '#4EC9B0',
            'blue': '#61AFEF',
            'purple': '#C678DD',
            'yellow': '#E5C07B',
            'orange': '#D19A66',
            'green': '#98C379',
            'lavender': '#B4A5D1',
            'error': '#E06C75',
        }

        # Load YAML override if provided
        if yaml_path and yaml_path.exists() and yaml:
            try:
                with open(yaml_path) as f:
                    config = yaml.safe_load(f)
                    self._load_palette_from_yaml(config)
            except Exception as e:
                print(f"Warning: Could not load color theme: {e}", file=sys.stderr)

        self._generate_ansi_codes()

    def _load_palette_from_yaml(self, config: dict):
        """Parse YAML structure and update palette."""
        if 'palette' not in config:
            return

        palette = config['palette']
        for category in ['background', 'foreground', 'accents', 'status']:
            if category in palette:
                for name, data in palette[category].items():
                    if isinstance(data, dict) and 'hex' in data:
                        key = f"{category[:2]}_{name}" if category in ['background', 'foreground'] else name
                        self._palette[key] = data['hex']

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_ansi_fg(self, r: int, g: int, b: int) -> str:
        """Generate ANSI 24-bit foreground color."""
        return f"\033[38;2;{r};{g};{b}m"

    def _rgb_to_ansi_bg(self, r: int, g: int, b: int) -> str:
        """Generate ANSI 24-bit background color."""
        return f"\033[48;2;{r};{g};{b}m"

    def _generate_ansi_codes(self):
        """Generate all ANSI color codes from palette."""
        for key, hex_val in self._palette.items():
            r, g, b = self._hex_to_rgb(hex_val)
            setattr(self, key.upper(), self._rgb_to_ansi_fg(r, g, b))

        # Background colors
        r, g, b = self._hex_to_rgb(self._palette['bg_selected'])
        self.BG_SELECTED = self._rgb_to_ansi_bg(r, g, b)

        # Common combinations
        self.CYAN_BOLD = self.CYAN + self.BOLD
        self.GREEN_BOLD = self.GREEN + self.BOLD
        self.BLUE_BOLD = self.BLUE + self.BOLD
        self.YELLOW_BOLD = self.YELLOW + self.BOLD
        self.ORANGE_BOLD = self.ORANGE + self.BOLD
        self.ERROR_BOLD = self.ERROR + self.BOLD
        self.PURPLE_BOLD = self.PURPLE + self.BOLD

        # Selection style
        self.SELECTION = self.BG_SELECTED + self.FG_PRIMARY

    def status_color(self, state: str) -> str:
        """Map torrent state to semantic color."""
        if state in STATE_DOWNLOAD:
            return self.CYAN_BOLD
        elif state in STATE_UPLOAD:
            return self.BLUE
        elif state in STATE_PAUSED:
            return self.FG_SECONDARY
        elif state in STATE_ERROR:
            return self.ERROR_BOLD
        elif state in STATE_CHECKING:
            return self.BLUE
        elif state in STATE_COMPLETED:
            return self.GREEN
        return self.FG_PRIMARY

LOCAL_TZ = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
PRESET_FILE = Path(__file__).parent.parent / "config" / "qbit-filter-presets.yml"
TRACKERS_LIST_URL = "https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_best.txt"
QC_TAG_TOOL = Path(__file__).resolve().parent / "media_qc_tag.py"
QC_LOG_DIR = Path.home() / ".logs" / "media_qc"
ACTIVE_QC_PROCESSES = {}  # hash -> (pid, start_time)

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
NEED_RESIZE = False

def handle_winch(signum, frame):
    global NEED_RESIZE
    NEED_RESIZE = True

def read_input_queue() -> list[str]:
    """Read all pending input and return a list of mapped keys."""
    keys = []
    while True:
        # Non-blocking check for input
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            break
            
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Peek for sequence
            seq = ""
            start = time.monotonic()
            while (time.monotonic() - start) < 0.15:
                if select.select([sys.stdin], [], [], 0.02)[0]:
                    c = sys.stdin.read(1)
                    seq += c
                    if seq.endswith(("A", "B", "C", "D", "H", "F", "Z", "~")):
                        break
                    if len(seq) > 1 and seq[-1].isalpha() and seq[-1] not in "O[": 
                         # Catch-all for other terminators
                        break
                else:
                    # No more data currently available
                    break
            
            if not seq:
                keys.append("ESC")
            elif seq in ("[Z", "[1;2Z", "[1;2I"):
                keys.append("SHIFT_TAB")
            elif seq.startswith("[1;5") or seq.startswith("[1;6"):
                if seq.endswith("I") or seq.endswith("Z"):
                    keys.append("CTRL_TAB")
            elif seq in ("[A", "[B", "[C", "[D"):
                # Arrows are intentionally ignored per requirements
                pass
            # We intentionally consume and drop other sequences (arrows, etc) to prevent leaking
        else:
            keys.append(ch)
    return keys

def get_key() -> str:
    # Deprecated compatibility wrapper if needed, but we will remove calls to it
    # For read_line, it expects to call input(), so we don't need this.
    # We'll leave a dummy or remove it.
    return ""



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
    except Exception as exc:
        return f"Error: {exc}"


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

    # Clean up completed processes
    for h in list(ACTIVE_QC_PROCESSES.keys()):
        pid, _ = ACTIVE_QC_PROCESSES[h]
        try:
            os.kill(pid, 0)  # Check if process exists (doesn't actually kill)
        except OSError:
            # Process doesn't exist, remove from tracking
            del ACTIVE_QC_PROCESSES[h]

    # Check if QC is already running for this hash
    if hash_value in ACTIVE_QC_PROCESSES:
        pid, start_time = ACTIVE_QC_PROCESSES[hash_value]
        elapsed = int(time.time() - start_time)
        return f"QC already running (PID {pid}, {elapsed}s ago)"

    QC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = QC_LOG_DIR / f"qc_tag_{hash_value[:8]}.log"
    cmd = [sys.executable, str(QC_TAG_TOOL), "--hash", hash_value, "--apply"]
    with log_path.open("a") as handle:
        handle.write(f"\n=== qc-tag-media {hash_value} @ {datetime.now(LOCAL_TZ).isoformat()} ===\n")
        handle.write(f"cmd={' '.join(cmd)}\n")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_path.open("a"),
            stderr=log_path.open("a"),
            start_new_session=True,
        )
        # Track this process
        ACTIVE_QC_PROCESSES[hash_value] = (proc.pid, time.time())
    except Exception as exc:
        return f"Failed ({exc})"
    return f"Queued (PID {proc.pid}, log: {log_path})"


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


# Legacy wrapper functions - will be removed after migration
# Note: These require colors to be initialized in main()
def status_color(state: str, colors_instance=None) -> str:
    """Deprecated: Use colors.status_color() instead."""
    if colors_instance:
        return colors_instance.status_color(state)
    # Fallback for backward compatibility during migration
    s = (state or "").strip()
    if s in STATE_ERROR:
        return "\033[38;2;224;108;117m"  # error
    if s in STATE_DOWNLOAD:
        return "\033[38;2;78;201;176m\033[1m"  # cyan_bold
    if s in STATE_UPLOAD:
        return "\033[38;2;97;175;239m"  # blue
    if s in STATE_PAUSED:
        return "\033[38;2;160;174;192m"  # fg_secondary
    if s in STATE_COMPLETED:
        return "\033[38;2;152;195;121m"  # green
    if s in STATE_CHECKING:
        return "\033[38;2;97;175;239m"  # blue
    return "\033[38;2;229;231;235m"  # fg_primary


def mode_color(mode: str, colors_instance=None) -> str:
    """Deprecated: Use colors directly instead."""
    if not colors_instance:
        # Fallback for backward compatibility during migration
        mode_map = {
            "i": "\033[96m",  # bright_cyan
            "p": "\033[93m",  # bright_yellow
            "d": "\033[91m",  # bright_red
            "c": "\033[38;5;214m",  # orange
            "t": "\033[38;5;210m",  # pink
            "v": "\033[94m",  # bright_blue
            "A": "\033[92m",  # bright_green
            "Q": "\033[95m",  # bright_purple
            "l": "\033[97m",  # bright_white
            "m": "\033[35m",  # magenta
        }
        return mode_map.get(mode, "\033[0m")

    # Use new color scheme
    mode_map = {
        "i": colors_instance.CYAN_BOLD,
        "p": colors_instance.YELLOW_BOLD,
        "d": colors_instance.ERROR_BOLD,
        "c": colors_instance.ORANGE,
        "t": colors_instance.PURPLE,
        "v": colors_instance.BLUE_BOLD,
        "A": colors_instance.GREEN_BOLD,
        "Q": colors_instance.PURPLE_BOLD,
        "l": colors_instance.FG_PRIMARY,
        "m": colors_instance.PURPLE,
    }
    return mode_map.get(mode, colors_instance.RESET)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(value: str) -> int:
    """Calculate display width accounting for ANSI codes and emoji."""
    plain = ANSI_RE.sub("", value)

    # Account for emoji and other wide characters
    # Most emoji and East Asian characters display as 2-wide
    width = 0
    for char in plain:
        code = ord(char)
        # Emoji ranges (simplified - covers most common emoji)
        if (0x1F300 <= code <= 0x1F9FF or  # Misc Symbols and Pictographs, Emoticons, etc.
            0x2600 <= code <= 0x26FF or    # Misc symbols (⚡ etc.)
            0x2700 <= code <= 0x27BF or    # Dingbats
            0xFE00 <= code <= 0xFE0F or    # Variation Selectors
            0x1F600 <= code <= 0x1F64F):   # Emoticons
            width += 2
        else:
            width += 1
    return width


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
MI_CACHE_MAX_ITEMS = 1000  # Limit queue to 1000 items to prevent memory leak
MI_CACHE_MAX_AGE_SECONDS = 86400  # 24 hours


def get_content_path(torrent_raw: dict) -> str:
    """Extract content path from torrent metadata."""
    content_path = torrent_raw.get("content_path")
    if not content_path:
        save_path = torrent_raw.get("save_path") or ""
        item_name = torrent_raw.get("name") or ""
        content_path = str(Path(save_path) / item_name) if save_path and item_name else ""
    return content_path


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


def get_mediainfo_summary_cached(hash_value: str, content_path: str, background_only: bool = False) -> str:
    cache_file = CACHE_DIR / f"{hash_value}.summary"
    if cache_file.exists():
        return cache_file.read_text().strip()
    if background_only:
        return "MI: loading..."
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


# ============================================================================
# HEADER & FOOTER REDESIGN (v1.7.0)
# ============================================================================

def draw_header_v2(
    colors: ColorScheme,
    api_url: str,
    version: str,
    torrents: list[dict],
    scope: str,
    sort_field: str,
    sort_desc: bool,
    page: int,
    total_pages: int,
    filters: list[dict],
    width: int
) -> list[str]:
    """
    Render professional 3-line header.

    Line 1: App branding + API endpoint + date
    Line 2: Stats dashboard with real-time bandwidth
    Line 3: Border
    """
    lines = []

    # Calculate stats
    downloading = sum(1 for t in torrents if t.get("state") in STATE_DOWNLOAD)
    seeding = sum(1 for t in torrents if t.get("state") in STATE_UPLOAD)
    paused = sum(1 for t in torrents if t.get("state") in STATE_PAUSED)
    completed = sum(1 for t in torrents if t.get("progress", 0) == 1.0)
    errors = sum(1 for t in torrents if t.get("state") in STATE_ERROR)

    # Calculate bandwidth
    total_dl = sum(t.get("dlspeed", 0) for t in torrents)
    total_ul = sum(t.get("upspeed", 0) for t in torrents)

    def fmt_speed(speed: int) -> str:
        if speed == 0: return "0"
        elif speed < 1024: return f"{speed} B/s"
        elif speed < 1024 * 1024: return f"{speed / 1024:.1f} KB/s"
        else: return f"{speed / (1024 * 1024):.1f} MB/s"

    # Line 1: Title bar (using visible_len for emoji-aware width calculation)
    left = f"{colors.CYAN_BOLD}⚡ QBITTORRENT TUI{colors.RESET} {colors.FG_SECONDARY}v{version}{colors.RESET}"
    center = f"{colors.FG_SECONDARY}📡 {colors.BLUE}{api_url}{colors.RESET}"
    right = f"{colors.FG_SECONDARY}⏰ {datetime.now().strftime('%Y-%m-%d')}{colors.RESET}"

    # Calculate spacing using visible_len (accounts for emoji width)
    left_visible = visible_len(left)
    center_visible = visible_len(center)
    right_visible = visible_len(right)

    total_visible = left_visible + center_visible + right_visible
    remaining = (width - 4) - total_visible  # -4 for "│ " and " │"

    if remaining > 0:
        left_pad = remaining // 2
        right_pad = remaining - left_pad
        title_line = f"{left}{' ' * left_pad}{center}{' ' * right_pad}{right}"
    else:
        title_line = f"{left}  {center}  {right}"

    lines.append(f"┌{'─' * (width - 2)}┐")
    lines.append(f"│ {title_line} │")
    lines.append(f"├{'─' * (width - 2)}┤")

    # Line 2: Stats dashboard
    stats = []

    # Real-time bandwidth (if active) - always show both for consistency
    if total_dl > 0 or total_ul > 0:
        stats.append(f"{colors.CYAN}⚡{colors.RESET}")
        dl_color = colors.CYAN_BOLD if total_dl > 0 else colors.FG_TERTIARY
        stats.append(f"{colors.CYAN}↓ {dl_color}{fmt_speed(total_dl)}{colors.RESET}")
        ul_color = colors.BLUE_BOLD if total_ul > 0 else colors.FG_TERTIARY
        stats.append(f"{colors.BLUE}↑ {ul_color}{fmt_speed(total_ul)}{colors.RESET}")
        stats.append(f"{colors.FG_SECONDARY}│{colors.RESET}")

    # Counts
    stats.append(f"{colors.CYAN}↓ {colors.CYAN_BOLD}{downloading}{colors.RESET}")
    stats.append(f"{colors.BLUE}↑ {colors.BLUE_BOLD}{seeding}{colors.RESET}")
    stats.append(f"{colors.FG_SECONDARY}⏸ {paused}{colors.RESET}")
    stats.append(f"{colors.GREEN}✓ {colors.GREEN_BOLD}{completed}{colors.RESET}")

    if errors > 0:
        stats.append(f"{colors.ERROR}✗ {colors.ERROR_BOLD}{errors}{colors.RESET}")
    else:
        stats.append(f"{colors.FG_TERTIARY}{colors.DIM}✗ 0{colors.RESET}")

    stats.append(f"{colors.FG_SECONDARY}│{colors.RESET}")

    # Scope/Sort/Pagination
    scope_display = scope.upper() if scope != "all" else "ALL"
    stats.append(f"{colors.FG_SECONDARY}Showing: {colors.YELLOW_BOLD}{scope_display}{colors.RESET}")
    stats.append(f"{colors.FG_SECONDARY}│{colors.RESET}")

    sort_arrow = "↓" if sort_desc else "↑"
    stats.append(f"{colors.FG_SECONDARY}Sort: {colors.YELLOW}{sort_field} {sort_arrow}{colors.RESET}")
    stats.append(f"{colors.FG_SECONDARY}│{colors.RESET}")

    stats.append(f"{colors.FG_SECONDARY}Pg {page + 1}/{total_pages}{colors.RESET}")

    # Active filters indicator
    if filters:
        active = [f for f in filters if f.get("enabled", True)]
        if active:
            stats.append(f"{colors.FG_SECONDARY}│{colors.RESET}")
            stats.append(f"{colors.ORANGE}[{len(active)} filters]{colors.RESET}")

    stats_line = "  ".join(stats)
    # Pad stats line to match border width
    stats_visible = visible_len(stats_line)
    padding_needed = (width - 4) - stats_visible  # -4 for "│ " and " │"
    if padding_needed > 0:
        stats_line += " " * padding_needed
    lines.append(f"│ {stats_line} │")
    lines.append(f"└{'─' * (width - 2)}┘")

    return lines


def draw_footer_v2(
    colors: ColorScheme,
    context: str,
    width: int,
    has_selection: bool = False
) -> list[str]:
    """
    Render context-sensitive grouped footer.

    Args:
        context: 'main', 'trackers', or 'mediainfo'
        width: Terminal width
        has_selection: Whether a torrent is selected
    """
    lines = []
    lines.append(f"┌{'─' * (width - 2)}┐")

    if context == "main":
        # Line 1: Actions
        actions = []

        if has_selection:
            actions.extend([
                f"{colors.CYAN_BOLD}P{colors.RESET}{colors.FG_SECONDARY}=Pause{colors.RESET}",
                f"{colors.CYAN_BOLD}V{colors.RESET}{colors.FG_SECONDARY}=Verify{colors.RESET}",
                f"{colors.CYAN_BOLD}C{colors.RESET}{colors.FG_SECONDARY}=Category{colors.RESET}",
                f"{colors.CYAN_BOLD}E{colors.RESET}{colors.FG_SECONDARY}=Tags{colors.RESET}",
                f"{colors.CYAN_BOLD}A{colors.RESET}{colors.FG_SECONDARY}=Trackers{colors.RESET}",
                f"{colors.CYAN_BOLD}Q{colors.RESET}{colors.FG_SECONDARY}=QC{colors.RESET}",
                f"{colors.ORANGE_BOLD}D{colors.RESET}{colors.FG_SECONDARY}=Delete{colors.RESET}",
            ])
        else:
            actions.append(f"{colors.FG_TERTIARY}{colors.DIM}(Select torrent for actions){colors.RESET}")

        actions_line = f"{colors.FG_SECONDARY}ACTIONS:{colors.RESET} " + "  ".join(actions)
        padding = width - visible_len(actions_line) - 4  # -4 for borders and spaces
        lines.append(f"│ {actions_line}{' ' * max(0, padding)} │")

        # Line 2: Navigation and View
        nav_parts = [
            f"{colors.YELLOW_BOLD}↑/↓{colors.RESET}{colors.FG_SECONDARY}=Move{colors.RESET}",
            f"{colors.YELLOW_BOLD}PgUp/Dn{colors.RESET}{colors.FG_SECONDARY}=Page{colors.RESET}",
            f"{colors.YELLOW_BOLD}Space{colors.RESET}{colors.FG_SECONDARY}=Select{colors.RESET}",
            f"{colors.BLUE_BOLD}Enter{colors.RESET}{colors.FG_SECONDARY}=Details{colors.RESET}",
            f"{colors.YELLOW_BOLD}Tab{colors.RESET}{colors.FG_SECONDARY}=Tabs{colors.RESET}",
        ]

        view_parts = [
            f"{colors.PURPLE_BOLD}?{colors.RESET}{colors.FG_SECONDARY}=Help{colors.RESET}",
            f"{colors.PURPLE_BOLD}q{colors.RESET}{colors.FG_SECONDARY}=Quit{colors.RESET}",
        ]

        nav_line = (
            f"{colors.FG_SECONDARY}NAVIGATE:{colors.RESET} " +
            "  ".join(nav_parts) +
            f"  {colors.FG_SECONDARY}│ VIEW:{colors.RESET} " +
            "  ".join(view_parts)
        )
        padding = width - visible_len(nav_line) - 4
        lines.append(f"│ {nav_line}{' ' * max(0, padding)} │")

    elif context == "trackers":
        title_line = f"{colors.CYAN_BOLD}TRACKER VIEW{colors.RESET}"
        padding = width - visible_len(title_line) - 4
        lines.append(f"│ {title_line}{' ' * max(0, padding)} │")

        actions = [
            f"{colors.CYAN_BOLD}A{colors.RESET}{colors.FG_SECONDARY}=Add{colors.RESET}",
            f"{colors.ORANGE_BOLD}D{colors.RESET}{colors.FG_SECONDARY}=Delete{colors.RESET}",
            f"{colors.CYAN_BOLD}E{colors.RESET}{colors.FG_SECONDARY}=Edit{colors.RESET}",
            f"{colors.CYAN_BOLD}R{colors.RESET}{colors.FG_SECONDARY}=Refresh{colors.RESET}",
        ]

        nav = [
            f"{colors.YELLOW_BOLD}↑/↓{colors.RESET}{colors.FG_SECONDARY}=Select{colors.RESET}",
            f"{colors.PURPLE_BOLD}Esc{colors.RESET}{colors.FG_SECONDARY}=Back{colors.RESET}",
            f"{colors.PURPLE_BOLD}?{colors.RESET}{colors.FG_SECONDARY}=Help{colors.RESET}",
        ]

        cmd_line = "  ".join(actions) + f"  {colors.FG_SECONDARY}│{colors.RESET}  " + "  ".join(nav)
        padding = width - visible_len(cmd_line) - 4
        lines.append(f"│ {cmd_line}{' ' * max(0, padding)} │")

    elif context == "mediainfo":
        title_line = f"{colors.LAVENDER}MEDIAINFO VIEW{colors.RESET}"
        padding = width - visible_len(title_line) - 4
        lines.append(f"│ {title_line}{' ' * max(0, padding)} │")

        actions = [
            f"{colors.CYAN_BOLD}Tab{colors.RESET}{colors.FG_SECONDARY}=Next{colors.RESET}",
            f"{colors.PURPLE_BOLD}Esc{colors.RESET}{colors.FG_SECONDARY}=Back{colors.RESET}",
            f"{colors.PURPLE_BOLD}?{colors.RESET}{colors.FG_SECONDARY}=Help{colors.RESET}",
        ]

        cmd_line = "  ".join(actions)
        padding = width - visible_len(cmd_line) - 4
        lines.append(f"│ {cmd_line}{' ' * max(0, padding)} │")

    lines.append(f"└{'─' * (width - 2)}┘")

    return lines


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


def format_filters_line(filters: list[dict], colors: ColorScheme) -> str:
    if not filters:
        return "Filters: -"
    parts = []
    for flt in filters:
        active = flt.get("enabled", True)
        color = colors.PURPLE if active else ""
        reset = colors.RESET if active else ""
        if flt["type"] == "text":
            prefix = "!" if flt.get("negate") else ""
            parts.append(f"text={color}{prefix}{flt['value']}{reset}")
        elif flt["type"] == "category":
            cat_color = colors.PURPLE if active else ""
            cat_reset = colors.RESET if active else ""
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


def render_mediainfo_lines(item: dict, width: int, colors: ColorScheme) -> list[str]:
    raw = item.get("raw") or {}
    content_path = get_content_path(raw)
    info = get_mediainfo_for_hash(str(item.get("hash") or ""), content_path)

    lines = []
    if not info:
        return [f"{colors.FG_TERTIARY}No MediaInfo.{colors.RESET}"]

    # Parse mediainfo output and colorize
    for line in str(info).splitlines():
        line = line.strip()
        if not line:
            lines.append("")
            continue

        # Check if it's a key: value line
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2:
                key, value = parts
                key = key.strip()
                value = value.strip()

                # Color label in lavender
                colored_key = f"{colors.LAVENDER}{key}:{colors.RESET}"

                # Color value based on type
                if any(unit in value.lower() for unit in ["kb/s", "mb/s", "gb/s", "gb", "mb", "kb", "bits"]):
                    colored_value = f"{colors.YELLOW}{value}{colors.RESET}"
                elif value.replace(".", "").replace("-", "").isdigit():
                    colored_value = f"{colors.YELLOW}{value}{colors.RESET}"
                elif "/" in value or "\\" in value or ":" in value:
                    colored_value = f"{colors.BLUE}{value}{colors.RESET}"
                else:
                    colored_value = f"{colors.FG_PRIMARY}{value}{colors.RESET}"

                colored_line = f"{colored_key} {colored_value}"
                lines.extend(wrap_ansi(colored_line, width))
            else:
                lines.extend(wrap_ansi(line, width))
        else:
            # Section headers or non-key-value lines
            lines.extend(wrap_ansi(line, width))

    return lines or [f"{colors.FG_TERTIARY}No MediaInfo.{colors.RESET}"]


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
    content_path = get_content_path(raw)
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
    parser.add_argument("--color-theme", type=Path, metavar='PATH', help='Path to YAML color theme file (overrides default colors)')
    args = parser.parse_args()

    # Initialize global color scheme
    colors = ColorScheme(yaml_path=args.color_theme)

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
            return 0
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    # Setup Resize Handler
    signal.signal(signal.SIGWINCH, handle_winch)

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
    last_key_debug = "-"
    need_redraw = True
    output_buffer = ""

    def cycle_tabs(direction: int = 1, exit_after_last: bool = False) -> None:
        nonlocal in_tab_view, active_tab, have_full_draw
        if not selection_hash:
            set_banner("Select an item.")
            return
        
        selected_row = next((r for r in page_rows if r.get("hash") == selection_hash), None)
        if not selected_row:
            set_banner("Selection moved off page.")
            return
            
        available = resolve_available_tabs(opener, api_url, selected_row)
        if not available:
            available = ["Info"]
            
        if not in_tab_view:
            in_tab_view = True
            # When entering, try to preserve last active tab if available, else first/last based on direction
            if direction > 0:
                current_label = tabs[active_tab]
                if current_label not in available:
                    active_tab = tabs.index(available[0])
            else:
                # Entering backward: jump to last available
                active_tab = tabs.index(available[-1])
            have_full_draw = False
            return

        current_label = tabs[active_tab]
        if current_label not in available:
            active_tab = tabs.index(available[0])
            return
            
        idx = available.index(current_label)
        
        if exit_after_last and direction > 0 and idx == len(available) - 1:
            in_tab_view = False
            active_tab = 0 # Reset for next entry
            have_full_draw = False
            return
            
        # Normal cycle
        new_idx = (idx + direction) % len(available)
        next_label = available[new_idx]
        active_tab = tabs.index(next_label)
        have_full_draw = False

    def print_at(row: int, text: str) -> None:
        tui_print(f"\033[{row};1H\033[2K{text}", end="")

    def tui_print(text: str = "", end: str = "\r\n") -> None:
        # Buffer the output instead of immediate write
        nonlocal output_buffer
        output_buffer += f"{text}{end}"

    def tui_flush() -> None:
        nonlocal output_buffer
        if output_buffer:
            sys.stdout.write(output_buffer)
            sys.stdout.flush()
            output_buffer = ""

    def build_list_block(page_rows_local: list[dict]) -> list[str]:
        lines: list[str] = []
        lines.append(header_line)
        lines.append(divider_line)
        for idx, item in enumerate(page_rows_local, 0):
            selected = selection_hash == item.get("hash")
            status_col = colors.status_color(item.get("state") or "")
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
                lines.append(f"{colors.SELECTION}{base_line}{colors.RESET}")
            else:
                focus_col = f"{colors.CYAN}{focus_marker}{colors.RESET}" if idx == focus_idx else " "
                base_line = (
                    f"{focus_col:<1} {idx:<2} {status_col}{st:<2}{colors.RESET} "
                    f"{status_col}{name:<{name_width}}{colors.RESET} "
                    f"{str(item.get('progress') or '-'): <6} "
                    f"{str(item.get('size') or '-'): <{size_width}} "
                    f"{str(item.get('dlspeed') or '-'): <8} "
                    f"{str(item.get('upspeed') or '-'): <8} "
                    f"{str(item.get('eta') or '-'): <6} "
                    f"{added_value}"
                    f"{colors.PURPLE}{cat_val:<{cat_width}}{colors.RESET} "
                    f"{hash_display:<{hash_width}}"
                )
                lines.append(base_line)

            if show_mediainfo_inline:
                raw_item = item.get("raw") or {}
                content_path = get_content_path(raw_item)
                # Use background_only=True to prevent blocking draw
                mi_summary = get_mediainfo_summary_cached(str(item.get("hash") or ""), content_path, background_only=True)
                indent = "     "
                lines.append(f"{indent}{colors.FG_TERTIARY}{mi_summary}{colors.RESET}")

            if show_tags:
                tags_raw = str(item.get("tags") or "").strip()
                if tags_raw:
                    tag_parts = []
                    for tag in [t.strip() for t in tags_raw.split(",") if t.strip()]:
                        if "FAIL" in tag.upper():
                            tag_parts.append(f"{colors.ERROR}{tag}{colors.RESET}")
                        elif "cross-seed" in tag.lower():
                            tag_parts.append(f"{colors.ORANGE}{tag}{colors.RESET}")
                        else:
                            tag_parts.append(f"{colors.PURPLE}{tag}{colors.RESET}")
                    tags_line = ", ".join(tag_parts)
                    indent = "     tags: "
                    width = max(40, term_w - len(indent))
                    for line in wrap_ansi(tags_line, width):
                        lines.append(indent + line)
        return lines

    def update_mediainfo_cache(rows_sorted: list[dict], page_rows_visible: list[dict]) -> bool:
        """Process one item from the MediaInfo queue. Returns True if cache was updated and item is visible."""
        nonlocal mi_bootstrap_done, mi_queue, mi_queue_index, mi_last_tick
        if not rows_sorted or not show_mediainfo_inline:
            return False
        now_tick = time.monotonic()
        if now_tick - mi_last_tick < 0.3: # Slower tick to reduce flicker
            return False
        mi_last_tick = now_tick

        if not mi_bootstrap_done:
            targets = []
            # Bootstrap priority: current page
            targets.extend(page_rows_visible)
            
            for item in targets:
                hash_value = item.get("hash") or ""
                if not hash_value: continue
                cache_file = CACHE_DIR / f"{hash_value}.summary"
                if not cache_file.exists() and hash_value not in mi_queue:
                    mi_queue.append(hash_value)
            mi_bootstrap_done = True

        if not mi_queue:
            # Fill queue with remaining items
            for item in rows_sorted:
                hash_value = item.get("hash") or ""
                if not hash_value: continue
                cache_file = CACHE_DIR / f"{hash_value}.summary"
                if not cache_file.exists() and hash_value not in mi_queue:
                    mi_queue.append(hash_value)
            mi_queue_index = 0

        # Prune queue if too large (prevent memory leak)
        if len(mi_queue) > MI_CACHE_MAX_ITEMS:
            mi_queue = mi_queue[-MI_CACHE_MAX_ITEMS:]
            mi_queue_index = min(mi_queue_index, len(mi_queue))

        # Periodic cache file cleanup (every 100 queue cycles)
        if mi_queue and mi_queue_index % 100 == 0:
            now = time.time()
            if CACHE_DIR.exists():
                for cache_file in CACHE_DIR.glob("*.summary"):
                    try:
                        if (now - cache_file.stat().st_mtime) > MI_CACHE_MAX_AGE_SECONDS:
                            cache_file.unlink()
                    except OSError:
                        pass

        if mi_queue:
            hash_value = mi_queue[mi_queue_index % len(mi_queue)]
            mi_queue_index += 1
            item = next((r for r in rows_sorted if r.get("hash") == hash_value), None)
            if item:
                raw_item = item.get("raw") or {}
                content_path = get_content_path(raw_item)
                # Perform the actual extraction
                get_mediainfo_summary(hash_value, content_path)
                
                # Only redraw if the item is visible on screen
                is_visible = any(r.get("hash") == hash_value for r in page_rows_visible)
                return is_visible
        return False

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            now = time.monotonic()
            data_changed = False
            if not cached_rows or (now - cache_time) >= fetch_interval:
                raw = qbit_request(opener, api_url, "GET", "/api/v2/torrents/info")
                if raw.startswith("Error:") or raw.startswith("HTTP "):
                    set_banner(f"Network error: {raw}")
                    cache_time = now
                else:
                    try:
                        torrents = json.loads(raw) if raw else []
                        rows = build_rows(torrents)
                        cached_torrents = torrents
                        cached_rows = rows
                        data_changed = True
                    except json.JSONDecodeError:
                        set_banner("Error: Invalid JSON response")
                    cache_time = now

            rows_to_render = cached_rows
            if scope != "all":
                rows_to_render = [r for r in rows_to_render if state_group(r.get("raw", {}).get("state", "")) == scope]

            rows_to_render = apply_filters(rows_to_render, filters)

            sort_field = sort_fields[sort_index]
            def sort_key(row: dict):
                raw = row.get("raw", {})
                if sort_field == "added_on": return raw.get("added_on") or 0
                if sort_field == "name": return row.get("name", "")
                if sort_field == "state": return state_group(raw.get("state", ""))
                if sort_field == "ratio": return raw.get("ratio") or 0
                if sort_field == "progress": return raw.get("progress") or 0
                if sort_field == "eta": return raw.get("eta") or 0
                if sort_field == "size": return raw.get("size") or raw.get("total_size") or 0
                if sort_field == "dlspeed": return raw.get("dlspeed") or 0
                if sort_field == "upspeed": return raw.get("upspeed") or 0
                return row.get("name", "")
            rows_to_render.sort(key=sort_key, reverse=sort_desc)
            
            page_rows, total_pages, page = format_rows(rows_to_render, page, args.page_size)

            if focus_idx >= len(page_rows):
                focus_idx = max(0, len(page_rows) - 1)

            # Update selection state
            selected_row_all = None
            if selection_hash:
                selected_row_all = next((r for r in cached_rows if r.get("hash") == selection_hash), None)
                if not selected_row_all:
                    selection_hash = selection_name = None
                    in_tab_view = False
                    set_banner("Selection cleared: item removed.")
                    data_changed = True
                else:
                    selection_name = selected_row_all.get("name") or selection_name

            # Background MediaInfo processing
            if update_mediainfo_cache(rows_to_render, page_rows):
                data_changed = True # Trigger redraw to show new MI data

            global NEED_RESIZE
            if NEED_RESIZE:
                term_w = terminal_width()
                have_full_draw = False
                need_redraw = True
                NEED_RESIZE = False

            if data_changed or need_redraw:
                term_w = terminal_width()
                banner_line = ""
                if banner_text and time.time() < banner_until:
                    banner_line = f"{colors.YELLOW_BOLD}{banner_text}{colors.RESET}"
                elif not selection_hash:
                    banner_line = f"{colors.FG_SECONDARY}Select an item.{colors.RESET}"
                else:
                    short_hash = (selection_hash or "")[:10]
                    banner_line = f"{colors.CYAN_BOLD}Selected:{colors.RESET} {selection_name} ({short_hash})"

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
                # Use content width for consistent layout
                content_width = min(len(header_line), term_w)
                divider_line = "-" * content_width
                list_block_lines = build_list_block(page_rows)

                if in_tab_view and selection_hash:
                    output_buffer = "\033[H\033[J" # Start with clear

                    # In tab view, use full terminal width for consistency
                    tab_display_width = term_w
                    content_width = tab_display_width  # Override for footer/dividers
                    divider_line = "-" * tab_display_width

                    # Use new v2 header
                    header_lines = draw_header_v2(
                        colors=colors,
                        api_url=api_url,
                        version=VERSION,
                        torrents=cached_torrents,
                        scope=scope,
                        sort_field=sort_fields[sort_index],
                        sort_desc=sort_desc,
                        page=page,
                        total_pages=total_pages,
                        filters=filters,
                        width=tab_display_width
                    )
                    for line in header_lines:
                        tui_print(line)

                    # Banner/selection info (if needed)
                    if banner_line:
                        tui_print(banner_line)
                        tui_print("")
                    selected_row = next((r for r in page_rows if r.get("hash") == selection_hash), None)
                    tab_divider = "-" * tab_display_width
                    if not selected_row:
                        tui_print("Selection not available on this page.")
                        tui_print(tab_divider)
                    else:
                        available_tabs = resolve_available_tabs(opener, api_url, selected_row)
                        if not available_tabs: available_tabs = ["Info"]
                        active_label = tabs[active_tab]
                        if active_label not in available_tabs:
                            active_tab = tabs.index(available_tabs[0])
                            active_label = tabs[active_tab]
                        tab_labels = []
                        for label in available_tabs:
                            if label == active_label:
                                tab_labels.append(f"{colors.YELLOW_BOLD}[{label}]{colors.RESET}")
                            else:
                                tab_labels.append(f"{colors.FG_SECONDARY}{label}{colors.RESET}")
                        tui_print("Tabs: " + " ".join(tab_labels))
                        tab_divider = "-" * tab_display_width
                        tui_print(tab_divider)
                        tab_width = max(40, tab_display_width - 4)
                        max_rows = max(10, shutil.get_terminal_size((100, 30)).lines - 15)
                        if active_label == "Info": content_lines = render_info_lines(selected_row, tab_width)
                        elif active_label == "Trackers":
                            trackers = fetch_trackers(opener, api_url, selection_hash)
                            content_lines = render_trackers_lines(trackers, tab_width, max_rows)
                        elif active_label == "Content":
                            files = fetch_files(opener, api_url, selection_hash)
                            content_lines = render_files_lines(files, tab_width, max_rows)
                        elif active_label == "Peers":
                            peers_payload = fetch_peers(opener, api_url, selection_hash)
                            content_lines = render_peers_lines(peers_payload, tab_width, max_rows)
                        else: content_lines = render_mediainfo_lines(selected_row, tab_width, colors)
                        for line in content_lines[:max_rows]: tui_print(line)
                        tui_print(tab_divider)
                        footer_row = 10 + len(content_lines[:max_rows]) + 1
                else:
                    if not have_full_draw:
                        output_buffer = "\033[H\033[J" # Start with clear
                        # Render new v2 header
                        header_lines = draw_header_v2(
                            colors=colors,
                            api_url=api_url,
                            version=VERSION,
                            torrents=cached_torrents,
                            scope=scope,
                            sort_field=sort_fields[sort_index],
                            sort_desc=sort_desc,
                            page=page,
                            total_pages=total_pages,
                            filters=filters,
                            width=content_width
                        )
                        for line in header_lines:
                            tui_print(line)

                        # Banner line (if any)
                        if banner_line:
                            tui_print(banner_line)
                            tui_print("")

                        # Torrent list
                        list_start_row = len(header_lines) + (2 if banner_line else 0)
                        for line in list_block_lines: tui_print(line)
                        tui_print(divider_line)
                        list_block_height = len(list_block_lines) + 1
                        footer_row = list_start_row + list_block_height
                        have_full_draw = True
                    else:
                        # For incremental updates, force full redraw to keep header in sync
                        # This ensures header stats are always current
                        have_full_draw = False
                        need_redraw = True
                        continue
                        print_at(row, divider_line)
                        list_block_height = current_height
                        footer_row = row + 1

                # Render Footer with absolute positioning
                row = footer_row

                # Show filters if any active
                if filters:
                    print_at(row, format_filters_line(filters, colors)); row += 1
                    print_at(row, divider_line); row += 1

                # Determine footer context
                footer_context = "main"
                if in_tab_view and selection_hash:
                    active_label = tabs[active_tab] if active_tab < len(tabs) else "Info"
                    if active_label == "Trackers":
                        footer_context = "trackers"
                    elif active_label == "MediaInfo":
                        footer_context = "mediainfo"

                # Render new v2 footer
                footer_lines = draw_footer_v2(
                    colors=colors,
                    context=footer_context,
                    width=content_width,
                    has_selection=bool(selection_hash)
                )

                for line in footer_lines:
                    print_at(row, line)
                    row += 1

                # Debug key display
                print_at(row, divider_line); row += 1
                print_at(row, f"Last Key: {colors.CYAN}{last_key_debug}{colors.RESET}\033[J")
                tui_flush()
                need_redraw = False

            # Wait for input or timeout
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not r:
                continue

            events = read_input_queue()
            for key in events:
                if not key: continue
                last_key_debug = key
                need_redraw = True

                if key == "\x11": return 0 # Ctrl-Q
                if key == "X":
                    if CACHE_DIR.exists():
                        shutil.rmtree(CACHE_DIR)
                        set_banner("MediaInfo cache cleared.")
                    continue
                if key == "?":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    sys.stdout.write("\033[H\033[J")
                    tui_print("Navigation: '(up) /(down) focus, , prev / . next page, Space select/clear, 0-9 select item")
                    tui_print("Scope: a=all, w=downloading, u=uploading, v=paused, e=completed, g=error")
                    tui_print("Sort: s=cycle field, o=toggle asc/desc")
                    tui_print("Filters: f=text, c=category, #=tag, l=line, x=manage stack, p=presets")
                    tui_print("Tabs: Tab=cycle tabs (off after last), Ctrl-Tab=cycle, T=cycle (selection required)")
                    tui_print("View: t=tags, d=added, h=hash width, m=inline mediainfo, X=clear mediainfo cache")
                    tui_print("Reset: z=default view (page 1, newest first)")
                    tui_print("Actions (selection required): P pause/resume, V verify, C category, E tags, A add trackers, Q qc, D delete")
                    tui_print("Esc clears selection (list) or exits tabs. Quit: Ctrl-Q")
                    tui_print("\nPress any key to continue...", end="")
                    sys.stdout.flush()
                    tty.setraw(fd)
                    read_input_queue() # Clear any pending
                    select.select([sys.stdin], [], [], 10.0) # Wait for a key
                    read_input_queue()
                    have_full_draw = False
                    continue
                if key == "`":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    sys.stdout.write("\033[H\033[J")
                    capture_key_sequences()
                    tty.setraw(fd)
                    have_full_draw = False
                    continue
                if key == "ESC":
                    if in_tab_view: in_tab_view = False; have_full_draw = False
                    elif selection_hash:
                        selection_hash = selection_name = None
                        set_banner("Selection cleared.")
                    continue
                if key == "CTRL_TAB":
                    cycle_tabs(direction=1, exit_after_last=False)
                    continue
                if key == "SHIFT_TAB":
                    cycle_tabs(direction=-1, exit_after_last=False)
                    continue
                if key == "'":
                    if page_rows: focus_idx = max(0, focus_idx - 1)
                    continue
                if key == "/":
                    if page_rows: focus_idx = min(len(page_rows) - 1, focus_idx + 1)
                    continue
                if key in (" ", "\r", "\n"):
                    if page_rows and 0 <= focus_idx < len(page_rows):
                        focused = page_rows[focus_idx]
                        if selection_hash and focused.get("hash") == selection_hash:
                            selection_hash = selection_name = None
                            set_banner("Selection cleared.")
                        else:
                            selection_hash = focused.get("hash")
                            selection_name = focused.get("name")
                            active_tab = 0 # Reset tab focus for new selection
                    continue
                if key == "\t":
                    cycle_tabs(direction=1, exit_after_last=True)
                    continue
                if key == "T":
                    cycle_tabs(direction=1, exit_after_last=False)
                    continue
                if key == "m":
                    show_mediainfo_inline = not show_mediainfo_inline
                    have_full_draw = False
                    continue
                if key == "z":
                    scope = "all"; page = 0; sort_index = 0; sort_desc = True; filters = []
                    show_tags = show_mediainfo_inline = show_full_hash = False
                    show_added = True; focus_idx = 0; in_tab_view = False; active_tab = 0
                    selection_hash = selection_name = None
                    have_full_draw = False
                    continue
                if key == ",":
                    if page > 0: page -= 1; focus_idx = 0
                    have_full_draw = False
                    continue
                if key == ".":
                    if page < total_pages - 1: page += 1; focus_idx = 0
                    have_full_draw = False
                    continue
                if key == "a": scope = "all"; page = 0; focus_idx = 0; have_full_draw = False; continue
                if key == "w": scope = "downloading"; page = 0; focus_idx = 0; have_full_draw = False; continue
                if key == "u": scope = "uploading"; page = 0; focus_idx = 0; have_full_draw = False; continue
                if key == "v": scope = "paused"; page = 0; focus_idx = 0; have_full_draw = False; continue
                if key == "e": scope = "completed"; page = 0; focus_idx = 0; have_full_draw = False; continue
                if key == "g": scope = "error"; page = 0; focus_idx = 0; have_full_draw = False; continue
                if key == "s": sort_index = (sort_index + 1) % len(sort_fields); have_full_draw = False; continue
                if key == "o": sort_desc = not sort_desc; have_full_draw = False; continue
                if key == "t": show_tags = not show_tags; have_full_draw = False; continue
                if key == "d": show_added = not show_added; have_full_draw = False; continue
                if key == "h": show_full_hash = not show_full_hash; have_full_draw = False; continue
                if key in "0123456789":
                    idx = int(key)
                    if idx < len(page_rows):
                        focus_idx = idx
                        selection_hash = page_rows[idx].get("hash")
                        selection_name = page_rows[idx].get("name")
                        active_tab = 0 # Reset tab focus for new selection
                    continue
                if key == "f":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    val = read_line("Text filter: ").strip()
                    if val:
                        negate = False
                        if val.startswith("!"): negate = True; val = val[1:]
                        filters = [f for f in filters if f.get("type") != "text"]
                        filters.append({"type": "text", "value": val, "enabled": True, "negate": negate})
                    tty.setraw(fd); have_full_draw = False; continue
                if key == "c":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    val = read_line("Category filter (blank clears, '-' for none): ").strip()
                    filters = [f for f in filters if f.get("type") != "category"]
                    if val:
                        negate = False
                        if val.startswith("!"): negate = True; val = val[1:]
                        filters.append({"type": "category", "value": val, "enabled": True, "negate": negate})
                    tty.setraw(fd); have_full_draw = False; continue
                if key == "#":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    val = read_line("Tag filter expression: ").strip()
                    filters = [f for f in filters if f.get("type") != "tag"]
                    if val:
                        parsed = parse_tag_filter(val)
                        if parsed: parsed["enabled"] = True; filters.append(parsed)
                    tty.setraw(fd); have_full_draw = False; continue
                if key == "l":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    val = read_line("Filter line (e.g. q=term cat=movies tag=tag1): ").strip()
                    if val: filters = parse_filter_line(val, filters)
                    tty.setraw(fd); have_full_draw = False; continue
                if key == "x":
                    if filters:
                        filters = [f for f in filters if not f.get("enabled", True)] if all(f.get("enabled", True) for f in filters) else [dict(f, enabled=True) for f in filters]
                    continue
                if key == "p":
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    tui_print("\nPresets (Slots):")
                    for s_idx in range(1, 10):
                        slot = presets.get(str(s_idx))
                        label = summarize_filters(restore_filters(slot)) if slot else "-"
                        tui_print(f"  {s_idx}: {label}")
                    val = read_line("\nSelect slot to load (1-9), or s[1-9] to save current: ").strip()
                    if val.isdigit() and val in presets:
                        filters = restore_filters(presets[val])
                        set_banner(f"Loaded slot {val}")
                    elif val.startswith("s") and val[1:].isdigit():
                        s_idx = val[1:]
                        presets[s_idx] = serialize_filters(filters)
                        save_presets(PRESET_FILE, presets)
                        set_banner(f"Saved current to slot {s_idx}")
                    tty.setraw(fd); have_full_draw = False; continue

                # Actions
                if selection_hash and key.upper() in "PVCEAQD":
                    selected_item = next((r for r in page_rows if r.get("hash") == selection_hash), None)
                    if selected_item:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        res = apply_action(opener, api_url, key.upper(), selected_item)
                        set_banner(f"Action {key.upper()}: {res}")
                        tty.setraw(fd); have_full_draw = False; continue

        return 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)