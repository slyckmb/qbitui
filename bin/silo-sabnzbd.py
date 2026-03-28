#!/usr/bin/env python3
"""Interactive SABnzbd dashboard with modes, hotkeys, and paging."""
import argparse
import json
import os
import select
import shutil
import sys
import readline  # Enables line editing for input()
import termios
import time
import tty
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

import silo_client_sab as _client

SCRIPT_NAME = "sabnzbd-dashboard"
VERSION = "1.2.0"
LAST_UPDATED = "2026-03-28"

COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_RED = "\033[31m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_GREY = "\033[90m"
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"


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


def terminal_width() -> int:
    try:
        return max(40, shutil.get_terminal_size((100, 20)).columns)
    except Exception:
        return 100


def read_api_key(path: Path) -> str:
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("SABNZBD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def read_api_url_from_config(path: Path) -> str:
    if not path.exists():
        return ""
    if yaml is not None:
        try:
            data = yaml.safe_load(path.read_text()) or {}
            api_url = (data.get("downloaders") or {}).get("sabnzbd", {}).get("api_url", "")
            if api_url:
                return api_url
        except Exception:
            pass

    api_url = ""
    in_downloaders = False
    in_sab = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            in_downloaders = line.strip() == "downloaders:"
            in_sab = False
            continue
        if in_downloaders and line.startswith("  ") and line.strip().endswith(":"):
            in_sab = line.strip() == "sabnzbd:"
            continue
        if in_downloaders and in_sab and line.strip().startswith("api_url:"):
            api_url = line.split("api_url:", 1)[1].strip()
            break
    return api_url


# Delegates to silo_client_sab — kept as shims for any external callers
def age_str(value) -> str:
    return _client._age_str(value)


def summarize(queue: dict, history: dict) -> str:
    return _client.summarize(queue, history)


def normalize_status(value: str) -> str:
    return str(value or "unknown").strip()


def status_color(status: str) -> str:
    s = (status or "").lower()
    if "fail" in s or "error" in s:
        return COLOR_RED
    if "pause" in s:
        return COLOR_CYAN
    if "queue" in s or "wait" in s:
        return COLOR_YELLOW
    if "download" in s or "repair" in s or "extract" in s or "propagating" in s:
        return COLOR_GREEN
    if "complete" in s:
        return COLOR_GREEN
    return COLOR_RESET


def build_rows(queue: dict, history: dict) -> list:
    rows = _client.build_rows_native(queue, history)
    # normalize_status applied for display consistency in this TUI
    for row in rows:
        row["status"] = normalize_status(row.get("status", ""))
    return rows


def format_rows(rows: list, page: int, page_size: int) -> list:
    total_pages = max(1, (len(rows) + page_size - 1) // page_size)
    page = max(0, min(page, total_pages - 1))
    start = page * page_size
    end = min(start + page_size, len(rows))
    return rows[start:end], total_pages, page


def apply_action(conn: "_client.SabConn", mode: str, item: dict) -> str:
    nzo_id = item.get("id")
    source = item.get("source")
    status = (item.get("status") or "").lower()

    if not nzo_id:
        return "Missing nzo_id"

    if mode == "p":
        if source != "Q":
            return "Pause/resume is only for queue items"
        action = "resume" if "pause" in status else "pause"
        result = _client.request(conn, {"mode": "queue", "name": action, "value": nzo_id})
        return "OK" if result.get("status") else (_client.last_error or "Failed")
    if mode == "d":
        api_mode = "history" if source == "H" else "queue"
        result = _client.request(conn, {"mode": api_mode, "name": "delete", "value": nzo_id})
        return "OK" if result.get("status") else (_client.last_error or "Failed")
    if mode == "c":
        print("Enter new category (blank cancels): ", end="", flush=True)
        value = input().strip()
        if not value:
            return "Cancelled"
        result = _client.request(conn, {"mode": "change_cat", "name": nzo_id, "value": value})
        return "OK" if result.get("status") else (_client.last_error or "Failed")
    if mode == "t":
        if source != "H":
            return "Retry is only for history items"
        result = _client.request(conn, {"mode": "retry", "value": nzo_id})
        return "OK" if result.get("status") else (_client.last_error or "Failed")
    if mode == "m":
        if source != "H":
            return "Mark-complete is only for history items"
        result = _client.request(conn, {"mode": "history", "name": "mark_as_completed", "value": nzo_id})
        return "OK" if result.get("status") else (_client.last_error or "Failed")
    return "Unknown mode"


def print_details(item: dict) -> None:
    raw = item.get("raw") or {}
    print(f"{COLOR_BOLD}Details{COLOR_RESET}")
    print(f"  Name: {item.get('name')}")
    print(f"  Status: {item.get('status')}")
    print(f"  Source: {item.get('source')}")
    print(f"  Category: {item.get('category')}")
    print(f"  Size: {item.get('size')}")
    print(f"  Progress: {item.get('progress')}")
    print(f"  ETA/Age: {item.get('eta_age')}")
    print(f"  ID: {item.get('id')}")
    for key in ("path", "storage", "message", "priority", "completed", "time_added"):
        if key in raw:
            print(f"  {key}: {raw.get(key)}")
    for key in ("fail_message", "fail_message_short", "fail_msg", "failmsg"):
        if key in raw and raw.get(key):
            print(f"  {key}: {raw.get(key)}")
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
    parser = argparse.ArgumentParser(description="Interactive SABnzbd dashboard")
    parser.add_argument("--config", default=os.environ.get("SABNZBD_CONFIG_FILE"), help="Path to request-cache.yml")
    parser.add_argument("--history-limit", type=int, default=int(os.environ.get("SABNZBD_HISTORY_LIMIT", "25")))
    parser.add_argument("--no-history", action="store_true", help="Disable history fetch")
    parser.add_argument("--page-size", type=int, default=int(os.environ.get("SABNZBD_PAGE_SIZE", "10")))
    args = parser.parse_args()
    show_history_env = os.environ.get("SABNZBD_SHOW_HISTORY", "").strip().lower()
    if show_history_env in ("0", "false", "no"):
        args.no_history = True

    config_path = Path(args.config) if args.config else (Path(__file__).parent.parent / "config" / "request-cache.yml")
    api_url = os.environ.get("SABNZBD_URL") or read_api_url_from_config(config_path) or "http://localhost:8080"

    api_key = os.environ.get("SABNZBD_API_KEY") or read_api_key(Path(os.environ.get("SABNZBD_API_KEY_FILE", "/mnt/config/secrets/sabnzbd/sabnzbd.env")))
    if not api_key:
        print("ERROR: SABNZBD_API_KEY not found (set SABNZBD_API_KEY or SABNZBD_API_KEY_FILE)", file=sys.stderr)
        return 1

    conn = _client.connect(api_url, api_key)

    mode = "i"
    scope = "all"
    page = 0
    filter_term = ""

    while True:
        queue = _client.request(conn, {"mode": "queue"}) or {}
        history = {}
        if not args.no_history:
            history = _client.request(conn, {"mode": "history", "limit": args.history_limit}) or {}
        rows = build_rows(queue, history)

        if scope == "queue":
            rows = [r for r in rows if r.get("source") == "Q"]
        elif scope == "history":
            rows = [r for r in rows if r.get("source") == "H"]

        if filter_term:
            f = filter_term.lower()
            rows = [r for r in rows if f in (r.get("name") or "").lower()]

        page_rows, total_pages, page = format_rows(rows, page, args.page_size)

        os.system("clear")
        print(f"{COLOR_BOLD}SABNZBD DASHBOARD (TUI){COLOR_RESET}")
        print(f"API: {conn.api_url}")
        if _client.last_error:
            print(f"{COLOR_RED}Error: {_client.last_error}{COLOR_RESET}")
        print(f"Summary: {summarize(queue, history)}")
        print("")

        scope_label = {"all": "ALL", "queue": "QUEUE", "history": "HISTORY"}[scope]
        mode_label = {"i": "INFO", "p": "PAUSE/RESUME", "d": "DELETE", "c": "CATEGORY", "t": "RETRY", "m": "MARK DONE"}[mode]
        filter_label = filter_term if filter_term else "-"
        page_label = f"Page {page + 1}/{total_pages}"
        print(f"Mode: {COLOR_BLUE}{mode_label}{COLOR_RESET}  Scope: {COLOR_MAGENTA}{scope_label}{COLOR_RESET}  Filter: {COLOR_GREY}{filter_label}{COLOR_RESET}  {page_label}")

        print("")
        print(f"{'No':<3} {'Src':<3} {'Status':<12} {'Name':<44} {'Prog':<7} {'Size':<10} {'ETA/Age':<8} {'Category':<12} ID")
        print("-" * max(80, terminal_width()))

        for idx, item in enumerate(page_rows, 1):
            color = status_color(item.get("status") or "")
            name = (item.get("name") or "")[:44]
            status = (item.get("status") or "")[:12]
            print(
                f"{idx:<3} {item.get('source', ''):<3} "
                f"{color}{status:<12}{COLOR_RESET} "
                f"{color}{name:<44}{COLOR_RESET} "
                f"{str(item.get('progress') or '-'): <7} "
                f"{str(item.get('size') or '-'): <10} "
                f"{str(item.get('eta_age') or '-'): <8} "
                f"{str(item.get('category') or '-'): <12} "
                f"{item.get('id') or ''}"
            )

        print("")
        print(
            "Keys: 1-9=Apply  r=Refresh  f=Filter  a=All  q=Queue  h=History  "
            "[/=Next ]=Prev  i/p/d/c/t/m=Mode  R=Raw  ?=Help  x=Quit"
        )

        key = get_key()
        if key in ("x", "\x1b"):
            break
        if key == "?":
            print("Modes: i=info, p=pause/resume, d=delete, c=change category, t=retry (history), m=mark done (history)")
            print("Paging: ] or / next page, [ previous page")
            print("Scope: a=all, q=queue, h=history  Raw: R + item number")
            print("Press any key to continue...", end="", flush=True)
            _ = get_key()
            continue
        if key == "R":
            print("\nRaw item number (blank cancels): ", end="", flush=True)
            raw_choice = input().strip()
            if raw_choice.isdigit():
                idx = int(raw_choice)
                if 1 <= idx <= len(page_rows):
                    print("")
                    print_raw(page_rows[idx - 1])
            continue
        if key == "r":
            continue
        if key == "a":
            scope = "all"
            page = 0
            continue
        if key == "q":
            scope = "queue"
            page = 0
            continue
        if key == "h":
            scope = "history"
            page = 0
            continue
        if key == "f":
            print("\nFilter (blank clears): ", end="", flush=True)
            filter_term = input().strip()
            page = 0
            continue
        if key in "ipdctm":
            mode = key
            continue
        if key == "[":
            page = total_pages - 1 if page == 0 else page - 1
            continue
        if key in ("]", "/"):
            page = 0 if page >= total_pages - 1 else page + 1
            continue
        if key and key.isdigit():
            idx = int(key)
            if 1 <= idx <= len(page_rows):
                item = page_rows[idx - 1]
                if mode == "i":
                    print("")
                    print_details(item)
                else:
                    if mode == "d":
                        print(f"Delete {item.get('name')}? (y/N): ", end="", flush=True)
                        confirm = input().strip().lower()
                        if confirm != "y":
                            continue
                    result = apply_action(conn, mode, item)
                    print(f"{mode_label}: {result}")
                    time.sleep(0.6)
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
