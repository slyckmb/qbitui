#!/usr/bin/env bash
# Version: 1.0.1
# rt-watch.sh — rTorrent state dashboard (port of qb-checking-watch.sh)
#
# Polls rTorrent via XMLRPC and displays torrent state counts.
# Env: RT_CONTAINER, RT_RPC_URL
set -euo pipefail

SCRIPT_VERSION="1.0.1"
RT_CONTAINER="${RT_CONTAINER:-rtorrent_vpn}"
RT_RPC_URL="${RT_RPC_URL:-http://localhost:8000/}"
INTERVAL_S=30
ONCE=0
UNTIL_CLEAR=0
MAX_ITERATIONS=0
DASHBOARD=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Watches rTorrent torrent state counts (checking/error/downloading/seeding).
Env: RT_CONTAINER (default: rtorrent_vpn), RT_RPC_URL (default: http://localhost:8000/)

States:
  checking    — hash check in progress (d.hashing>0)
  error       — tracker error message set (d.message non-empty)
  downloading — active download (state=1, incomplete, down.rate>0)
  stalledDL   — stalled download (state=1, incomplete, down.rate=0)
  seeding(up) — complete and started (uploading + stalledUP)
  stalledUP   — seeding, no upload activity
  uploading   — seeding, actively uploading
  stoppedUP   — complete, stopped
  stoppedDL   — incomplete, stopped

Options:
  --interval N        Poll interval seconds (default: 30)
  --once              Run one sample then exit
  --until-clear       Exit when checking=0 and downloading=0
  --max-iterations N  Exit after N polling iterations (0=infinite)
  --dashboard         Overwrite-in-place dashboard mode (like watch)
  -h, --help          Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --interval)    INTERVAL_S="${2:-}";       shift 2 ;;
  --once)        ONCE=1;                    shift   ;;
  --until-clear) UNTIL_CLEAR=1;            shift   ;;
  --max-iterations) MAX_ITERATIONS="${2:-}"; shift 2 ;;
  --dashboard)   DASHBOARD=1;              shift   ;;
  -h|--help)     usage; exit 0 ;;
  *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$INTERVAL_S"      =~ ^[0-9]+$ ]] || [[ "$INTERVAL_S" -lt 1 ]]; then
  echo "--interval must be a positive integer" >&2; exit 2
fi
if ! [[ "$MAX_ITERATIONS"  =~ ^[0-9]+$ ]]; then
  echo "--max-iterations must be a non-negative integer" >&2; exit 2
fi

_on_exit() {
  if [[ "$DASHBOARD" -eq 1 ]]; then printf '\n'; fi
}
trap '_on_exit' EXIT

if [[ "$DASHBOARD" -eq 0 ]]; then
  echo "watchdog_config interval_s=${INTERVAL_S} once=${ONCE} until_clear=${UNTIL_CLEAR} max_iterations=${MAX_ITERATIONS} container=${RT_CONTAINER}"
fi

# ── rTorrent XMLRPC fetch ─────────────────────────────────────────────────────
# Calls d.multicall2 via host-side XMLRPC first, then docker exec fallback.
# Returns JSON array of torrent objects.
# Fields per torrent: hash, state (derived string), progress, down_rate, up_rate, message, label
_rt_fetch_json() {
  python3 - "$RT_CONTAINER" "$RT_RPC_URL" <<'PYEOF'
import sys, subprocess, re, json

container = sys.argv[1]
rpc_url   = sys.argv[2]

body = (
    '<?xml version="1.0"?>'
    '<methodCall><methodName>d.multicall2</methodName><params>'
    '<param><value><string></string></value></param>'   # target (empty = default view)
    '<param><value><string>main</string></value></param>'
    '<param><value><string>d.hash=</string></value></param>'
    '<param><value><string>d.state=</string></value></param>'
    '<param><value><string>d.hashing=</string></value></param>'
    '<param><value><string>d.complete=</string></value></param>'
    '<param><value><string>d.down.rate=</string></value></param>'
    '<param><value><string>d.up.rate=</string></value></param>'
    '<param><value><string>d.message=</string></value></param>'
    '<param><value><string>d.custom1=</string></value></param>'
    '<param><value><string>d.size_bytes=</string></value></param>'
    '<param><value><string>d.completed_bytes=</string></value></param>'
    '</params></methodCall>'
)

def run_fetch(cmd, timeout):
    try:
        result = subprocess.run(
            cmd,
            input=body,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout

xml = run_fetch(
    ['curl', '-sf', '--max-time', '10', rpc_url, '-H', 'Content-Type: text/xml', '--data', '@-'],
    timeout=15,
)
if xml is None:
    xml = run_fetch(
        ['docker', 'exec', '-i', container,
         'curl', '-sf', '--max-time', '10',
         rpc_url, '-H', 'Content-Type: text/xml', '--data', '@-'],
        timeout=15,
    )
if xml is None:
    print('[]')
    sys.exit(0)

# Parse each per-torrent inner <array><data>...</data></array>
# d.multicall2 returns: methodResponse > params > param > value > array > data >
#   value (per torrent) > array > data > value (per field)
def _val(v):
    """Extract typed value from a single <value>...</value> inner content string."""
    m = re.search(r'<(?:i4|i8|int)>(\d+)</(?:i4|i8|int)>', v)
    if m:
        return int(m.group(1))
    m = re.search(r'<string>(.*?)</string>', v, re.DOTALL)
    if m:
        return m.group(1)
    # bare string (no type tag)
    stripped = re.sub(r'<[^>]+>', '', v).strip()
    return stripped

def parse_multicall(xml):
    torrents = []
    # Each torrent is one <value><array><data>...</data></array></value> inside the outer data
    for torrent_block in re.findall(
        r'<array>\s*<data>(.*?)</data>\s*</array>', xml, re.DOTALL
    ):
        fields = []
        for vblock in re.findall(r'<value>(.*?)</value>', torrent_block, re.DOTALL):
            fields.append(_val(vblock))
        if fields:
            torrents.append(fields)
    return torrents

def derive_state(state, hashing, complete, down_rate, up_rate, message):
    if hashing > 0:
        return 'checking'
    if message and message.strip():
        return 'error'
    if state == 0:
        return 'stoppedUP' if complete == 1 else 'stoppedDL'
    # state == 1 (started/active)
    if complete == 0:
        return 'downloading' if down_rate > 0 else 'stalledDL'
    # complete == 1, started
    return 'uploading' if up_rate > 0 else 'stalledUP'

raw = parse_multicall(xml)
output = []
for fields in raw:
    if len(fields) < 10:
        continue
    hash_     = str(fields[0])
    state     = int(fields[1]) if isinstance(fields[1], int) else 0
    hashing   = int(fields[2]) if isinstance(fields[2], int) else 0
    complete  = int(fields[3]) if isinstance(fields[3], int) else 0
    down_rate = int(fields[4]) if isinstance(fields[4], int) else 0
    up_rate   = int(fields[5]) if isinstance(fields[5], int) else 0
    message   = str(fields[6]) if fields[6] is not None else ''
    label     = str(fields[7]) if fields[7] is not None else ''
    size      = int(fields[8]) if isinstance(fields[8], int) else 0
    done      = int(fields[9]) if isinstance(fields[9], int) else 0
    progress  = (done / size) if size > 0 else (1.0 if complete else 0.0)

    output.append({
        'hash':      hash_.lower(),
        'state':     derive_state(state, hashing, complete, down_rate, up_rate, message),
        'progress':  progress,
        'down_rate': down_rate,
        'up_rate':   up_rate,
        'message':   message,
        'label':     label,
    })

print(json.dumps(output))
PYEOF
}

# ── Dashboard renderer ────────────────────────────────────────────────────────
print_dashboard() {
  local ts="$1"
  local checking="$2"
  local error="$3"
  local down="$4"
  local stalled_dl="$5"
  local seeding_up="$6"
  local stalled_up="$7"
  local uploading="$8"
  local stopped_up="$9"
  local stopped_dl="${10}"
  local total="${11}"
  local interval_s="${12}"

  local sep="──────────────────────"

  printf '── rTorrent v%s ──\n' "$SCRIPT_VERSION"
  printf '%s\n' "$ts"
  printf '%s\n' "$sep"
  printf '%-12s : %5s\n' "checking"    "$checking"
  printf '%-12s : %5s\n' "error"       "$error"
  printf '%-12s : %5s\n' "downloading" "$down"
  printf '%-12s : %5s\n' "stalledDL"   "$stalled_dl"
  printf '%-12s : %5s\n' "seeding(up)" "$seeding_up"
  printf '%-12s : %5s\n' "stalledUP"   "$stalled_up"
  printf '%-12s : %5s\n' "uploading"   "$uploading"
  printf '%-12s : %5s\n' "stoppedUP"   "$stopped_up"
  printf '%-12s : %5s\n' "stoppedDL"   "$stopped_dl"
  printf '%s\n' "$sep"
  printf '%-12s : %5s\n' "total"       "$total"
  printf '%s\n' "$sep"
  printf '%-12s : %4ss\n' "interval"   "$interval_s"
}

_FORCE_REDRAW=0
_on_winch() { _FORCE_REDRAW=1; }
[[ "$DASHBOARD" -eq 1 ]] && trap '_on_winch' SIGWINCH

iteration=0

while true; do
  iteration=$((iteration + 1))

  FETCH_ERROR=""
  TORRENTS_JSON="[]"
  _raw=""
  if ! _raw="$(_rt_fetch_json 2>/dev/null)"; then
    FETCH_ERROR="fetch_failed"
  elif [[ -z "$_raw" ]] || ! jq -e . >/dev/null 2>&1 <<<"$_raw"; then
    FETCH_ERROR="invalid_json"
  else
    TORRENTS_JSON="$_raw"
  fi

  if [[ -n "$FETCH_ERROR" ]]; then
    _err_ts="$(date '+%F %T')"
    if [[ "$DASHBOARD" -eq 1 ]]; then
      printf '\033[2J\033[H'
      printf '── rTorrent v%s ── ERROR\n' "$SCRIPT_VERSION"
      printf '%s\n' "$_err_ts"
      printf '──────────────────────\n'
      printf '%-12s : %s\n' "error" "$FETCH_ERROR"
      printf '──────────────────────\n'
      printf '%-12s : %4ss\n' "interval" "$INTERVAL_S"
      _FORCE_REDRAW=0
    else
      printf '%s error fetch_error=%s\n' "$_err_ts" "$FETCH_ERROR"
    fi
    if [[ "$ONCE" -eq 1 ]]; then exit 1; fi
    sleep "$INTERVAL_S" &
    _SLEEP_PID=$!
    wait "$_SLEEP_PID" 2>/dev/null || true
    kill "$_SLEEP_PID" 2>/dev/null || true
    continue
  fi

  read -r CHECKING ERROR DOWN STALLED_DL SEEDING_UP STALLED_UP UPLOADING STOPPED_UP STOPPED_DL TOTAL TOP_STATES <<<"$(jq -r '
    [
      ([.[] | select(.state == "checking")]    | length),
      ([.[] | select(.state == "error")]       | length),
      ([.[] | select(.state == "downloading")] | length),
      ([.[] | select(.state == "stalledDL")]   | length),
      ([.[] | select(.state == "uploading" or .state == "stalledUP")] | length),
      ([.[] | select(.state == "stalledUP")]   | length),
      ([.[] | select(.state == "uploading")]   | length),
      ([.[] | select(.state == "stoppedUP")]   | length),
      ([.[] | select(.state == "stoppedDL")]   | length),
      (length),
      (
        group_by(.state)
        | map({s: .[0].state, c: length})
        | sort_by(-.c)
        | .[:8]
        | map("\(.s)=\(.c)")
        | join(",")
      )
    ] | @tsv
  ' <<<"$TORRENTS_JSON")"

  if [[ "$DASHBOARD" -eq 1 ]]; then
    printf '\033[2J\033[H'
    _FORCE_REDRAW=0
    print_dashboard \
      "$(date '+%F %T')" \
      "$CHECKING" "$ERROR" "$DOWN" "$STALLED_DL" \
      "$SEEDING_UP" "$STALLED_UP" "$UPLOADING" \
      "$STOPPED_UP" "$STOPPED_DL" \
      "$TOTAL" "$INTERVAL_S"
  else
    printf '%s checking=%s error=%s downloading=%s stalledDL=%s seeding=%s stalledUP=%s uploading=%s stoppedUP=%s stoppedDL=%s total=%s top=%s\n' \
      "$(date '+%F %T')" \
      "$CHECKING" "$ERROR" "$DOWN" "$STALLED_DL" \
      "$SEEDING_UP" "$STALLED_UP" "$UPLOADING" \
      "$STOPPED_UP" "$STOPPED_DL" \
      "$TOTAL" "$TOP_STATES"
  fi

  if [[ "$UNTIL_CLEAR" -eq 1 && "$CHECKING" -eq 0 && "$DOWN" -eq 0 && "$STALLED_DL" -eq 0 ]]; then
    [[ "$DASHBOARD" -eq 0 ]] && echo "done checking=0 downloading=0 stalledDL=0"
    exit 0
  fi

  if [[ "$ONCE" -eq 1 ]]; then exit 0; fi
  if [[ "$MAX_ITERATIONS" -gt 0 && "$iteration" -ge "$MAX_ITERATIONS" ]]; then
    [[ "$DASHBOARD" -eq 0 ]] && echo "done max_iterations=${MAX_ITERATIONS}"
    exit 0
  fi

  sleep "$INTERVAL_S" &
  _SLEEP_PID=$!
  wait "$_SLEEP_PID" 2>/dev/null || true
  kill "$_SLEEP_PID" 2>/dev/null || true
done
