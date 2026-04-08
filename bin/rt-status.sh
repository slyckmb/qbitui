#!/usr/bin/env bash
# Version: 1.1.0
# rt-watch.sh — rTorrent state dashboard
#
# Reads rTorrent state from the shared silo cache by default.
# Direct XMLRPC access is available only via --direct for one-off diagnostics.
set -euo pipefail

SCRIPT_VERSION="1.2.2"
RT_CONTAINER="${RT_CONTAINER:-rtorrent_vpn}"
RT_RPC_URL="${RT_RPC_URL:-http://localhost:8000/}"
RT_CACHE_SUMMARY="${RT_CACHE_SUMMARY:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rt-cache-summary.py}"
RT_CACHE_DAEMON="${RT_CACHE_DAEMON:-/home/michael/dev/tools/silo/bin/silo-rt-cache-daemon.py}"
RT_CACHE_PYTHON="${RT_CACHE_PYTHON:-python3}"
RT_CACHE_MAX_AGE="${RT_CACHE_MAX_AGE:-60}"
INTERVAL_S=30
ONCE=0
UNTIL_CLEAR=0
MAX_ITERATIONS=0
DASHBOARD=0
DIRECT_MODE=0

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Watches rTorrent torrent state counts (checking/error/downloading/seeding).
Default transport is the shared silo RT cache. Use --direct only for one-off
diagnostics when you intentionally want to hit RT XMLRPC.

Env:
  RT_CACHE_SUMMARY  Cache summary helper path
  RT_CACHE_DAEMON   silo-rt-cache-daemon path
  RT_CACHE_PYTHON   Python interpreter for cache daemon/helper
  RT_CACHE_MAX_AGE  Fresh-cache threshold in seconds (default: 60)
  RT_CONTAINER      rTorrent container name for --direct fallback
  RT_RPC_URL        RT XMLRPC URL for --direct mode (default: http://localhost:8000/)

States:
  checking    — hash check in progress (d.hashing>0)
  error       — fatal: d.message set on incomplete item (download stuck/broken)
  trk_warn    — informational: d.message set on complete seeding item (tracker
                announce rejection; content is fine — peer_limit, multi_location, etc.)
  downloading — active download (state=1, incomplete, down.rate>0)
  stalledDL   — stalled download (state=1, incomplete, down.rate=0)
  seeding(up) — complete and started (uploading + stalledUP)
  stalledUP   — seeding, no upload activity
  uploading   — seeding, actively uploading
  stoppedUP   — complete, stopped
  stoppedDL   — incomplete, stopped

  Note: cache mode maps .states.error → trk_warn (pre-split; error_fatal=0 until
  silo cache daemon emits .states.error_fatal separately).

Options:
  --interval N        Poll interval seconds (default: 30)
  --once              Run one sample then exit
  --until-clear       Exit when checking=0 and downloading=0
  --max-iterations N  Exit after N polling iterations (0=infinite)
  --dashboard         Overwrite-in-place dashboard mode (like watch)
  --direct            Bypass cache and query RT XMLRPC directly
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
  --direct)      DIRECT_MODE=1;            shift   ;;
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
  echo "watchdog_config interval_s=${INTERVAL_S} once=${ONCE} until_clear=${UNTIL_CLEAR} max_iterations=${MAX_ITERATIONS} direct=${DIRECT_MODE} container=${RT_CONTAINER}"
fi

# ── Direct RT XMLRPC fetch (explicit only) ────────────────────────────────────
_rt_fetch_json_direct() {
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
    if state == 0:
        return 'stoppedUP' if complete == 1 else 'stoppedDL'
    # state == 1 (started/active)
    if complete == 0:
        # incomplete + message = download is stuck/broken (fatal)
        return 'error' if (message and message.strip()) else ('downloading' if down_rate > 0 else 'stalledDL')
    # complete == 1: item is seeding regardless of tracker announce result
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

# ── Shared silo RT cache summary ──────────────────────────────────────────────
_rt_fetch_cache_summary() {
  "$RT_CACHE_PYTHON" "$RT_CACHE_SUMMARY" \
    --daemon "$RT_CACHE_DAEMON" \
    --python "$RT_CACHE_PYTHON" \
    --max-age "$RT_CACHE_MAX_AGE"
}

# ── Dashboard renderer ────────────────────────────────────────────────────────
print_dashboard() {
  local ts="$1"
  local checking="$2"
  local error="$3"
  local trk_warn="$4"
  local trk_breakdown="$5"
  local down="$6"
  local stalled_dl="$7"
  local seeding_up="$8"
  local stalled_up="$9"
  local uploading="${10}"
  local stopped_up="${11}"
  local stopped_dl="${12}"
  local total="${13}"
  local interval_s="${14}"

  local sep="────────────────────────"

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
  if [[ "$trk_warn" -gt 0 ]] 2>/dev/null; then
    printf '%s\n' "$sep"
    printf '%-12s : %5s\n' "trk_warn"  "$trk_warn"
    if [[ -n "$trk_breakdown" ]]; then
      while IFS=$'\t' read -r cat cnt; do
        [[ -z "$cat" ]] && continue
        printf '  %-10s : %5s\n' "$cat" "$cnt"
      done <<<"$trk_breakdown"
    fi
  fi
}

_FORCE_REDRAW=0
_on_winch() { _FORCE_REDRAW=1; }
[[ "$DASHBOARD" -eq 1 ]] && trap '_on_winch' SIGWINCH

iteration=0

while true; do
  iteration=$((iteration + 1))

  FETCH_ERROR=""
  RT_TRANSPORT_LABEL=""
  RT_TRANSPORT_DETAIL=""
  CHECKING=0
  ERROR=0
  TRACKER_WARN=0
  TRACKER_WARN_BREAKDOWN=""
  DOWN=0
  STALLED_DL=0
  SEEDING_UP=0
  STALLED_UP=0
  UPLOADING=0
  STOPPED_UP=0
  STOPPED_DL=0
  TOTAL=0
  TOP_STATES=""

  if [[ "$DIRECT_MODE" -eq 1 ]]; then
    TORRENTS_JSON="[]"
    _raw=""
    if ! _raw="$(_rt_fetch_json_direct 2>/dev/null)"; then
      FETCH_ERROR="fetch_failed"
    elif [[ -z "$_raw" ]] || ! jq -e . >/dev/null 2>&1 <<<"$_raw"; then
      FETCH_ERROR="invalid_json"
    else
      TORRENTS_JSON="$_raw"
      RT_TRANSPORT_LABEL="direct"
      RT_TRANSPORT_DETAIL="xmlrpc"
      read -r CHECKING ERROR DOWN STALLED_DL SEEDING_UP STALLED_UP UPLOADING STOPPED_UP STOPPED_DL TOTAL TOP_STATES <<<"$(jq -r '
        [
          ([.[] | select(.state == "checking")]                             | length),
          ([.[] | select(.state == "error")]                                | length),
          ([.[] | select(.state == "downloading")]                          | length),
          ([.[] | select(.state == "stalledDL")]                            | length),
          ([.[] | select(.state == "uploading" or .state == "stalledUP")]   | length),
          ([.[] | select(.state == "stalledUP")]                            | length),
          ([.[] | select(.state == "uploading")]                            | length),
          ([.[] | select(.state == "stoppedUP")]                            | length),
          ([.[] | select(.state == "stoppedDL")]                            | length),
          (length),
          (group_by(.state) | map({s: .[0].state, c: length}) | sort_by(-.c)
           | .[:8] | map("\(.s)=\(.c)") | join(","))
        ] | @tsv
      ' <<<"$TORRENTS_JSON")"
      # Tracker warn breakdown: complete items with non-empty d.message (parallel count, not a state)
      TRACKER_WARN=0
      TRACKER_WARN_BREAKDOWN=""
      _twraw="$(jq -r '
        def cat:
          if test("already have [38] peer|Sorry max peers|max peers reached") then "peer_lim"
          elif test("3 location|rate limit.*location") then "multi_loc"
          elif test("SSL|certificate|SSL peer") then "ssl_err"
          elif test("Timeout|Host not found|stream truncat|Connection reset|non-authorit") then "conn_err"
          elif test("has been deleted") then "deleted"
          elif test("passkey|InfoHash|not found in [Hh]istory|Torrent not found") then "auth_err"
          else "other"
          end;
        [.[] | select(.message != "" and .state != "error")]
        | (length | tostring),
          (group_by(.message | cat)
           | map({k: (.[0].message | cat), n: length})
           | sort_by(-.n)[]
           | "\(.k)\t\(.n)")
      ' <<<"$TORRENTS_JSON" 2>/dev/null)" || true
      if [[ -n "$_twraw" ]]; then
        TRACKER_WARN="$(printf '%s' "$_twraw" | head -1)"
        TRACKER_WARN_BREAKDOWN="$(printf '%s' "$_twraw" | tail -n +2)"
      fi
    fi
  else
    SUMMARY_JSON=""
    if ! SUMMARY_JSON="$(_rt_fetch_cache_summary 2>/dev/null)"; then
      FETCH_ERROR="cache_summary_failed"
    elif [[ -z "$SUMMARY_JSON" ]] || ! jq -e . >/dev/null 2>&1 <<<"$SUMMARY_JSON"; then
      FETCH_ERROR="invalid_cache_summary"
    else
      RT_TRANSPORT_LABEL="cache"
      RT_TRANSPORT_DETAIL="$(jq -r '
        ((.cache_age_s // "?") | tostring | split(".")[0]) + "s" +
        (if .freshness != "fresh" then "·" + (.freshness // "?") else "" end) +
        (if .daemon_running == false then "·dmn↓" else "" end)
      ' <<<"$SUMMARY_JSON")"
      if [[ "$(jq -r '.ok' <<<"$SUMMARY_JSON")" != "true" ]]; then
        FETCH_ERROR="$(jq -r '.last_error // .status_error // "cache_missing"' <<<"$SUMMARY_JSON")"
      else
        read -r CHECKING ERROR DOWN STALLED_DL STALLED_UP UPLOADING STOPPED_UP STOPPED_DL TOTAL TOP_STATES <<<"$(jq -r '
          [
            ((.states.checkingDL // 0) + (.states.checkingUP // 0) + (.states.checking // 0)),
            (.states.error_fatal // 0),
            (.states.downloading // 0),
            (.states.stalledDL // 0),
            (.states.stalledUP // 0),
            (.states.uploading // 0),
            (.states.stoppedUP // 0),
            (.states.stoppedDL // 0),
            (.items // 0),
            ((.top_states // []) | join(","))
          ] | @tsv
        ' <<<"$SUMMARY_JSON")"
        # .states.error = old pre-split bucket = all d.message items = tracker warns (seeding)
        # Add them into seeding counts until silo splits the bucket
        TRACKER_WARN="$(jq -r '.states.error // 0' <<<"$SUMMARY_JSON")"
        TRACKER_WARN_BREAKDOWN=""
        SEEDING_UP=$((STALLED_UP + UPLOADING + TRACKER_WARN))
        STALLED_UP=$((STALLED_UP + TRACKER_WARN))
      fi
    fi
  fi

  if [[ -n "$FETCH_ERROR" ]]; then
    _err_ts="$(date '+%F %T')"
    if [[ "$DASHBOARD" -eq 1 ]]; then
      printf '\033[2J\033[H'
      printf '── rTorrent v%s ── ERROR\n' "$SCRIPT_VERSION"
      printf '%s\n' "$_err_ts"
      printf '────────────────────────\n'
      printf '%-12s : %s\n' "fetch_error" "$FETCH_ERROR"
      printf '────────────────────────\n'
      printf '%-12s : %5s\n' "total" "-"
      printf '────────────────────────\n'
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

  if [[ "$DASHBOARD" -eq 1 ]]; then
    printf '\033[2J\033[H'
    _FORCE_REDRAW=0
    print_dashboard \
      "$(date '+%F %T')" \
      "$CHECKING" "$ERROR" "$TRACKER_WARN" "$TRACKER_WARN_BREAKDOWN" \
      "$DOWN" "$STALLED_DL" \
      "$SEEDING_UP" "$STALLED_UP" "$UPLOADING" \
      "$STOPPED_UP" "$STOPPED_DL" \
      "$TOTAL" "$INTERVAL_S"
  else
    printf '%s transport=%s detail=%q checking=%s error=%s trk_warn=%s downloading=%s stalledDL=%s seeding=%s stalledUP=%s uploading=%s stoppedUP=%s stoppedDL=%s total=%s top=%s\n' \
      "$(date '+%F %T')" \
      "$RT_TRANSPORT_LABEL" "$RT_TRANSPORT_DETAIL" \
      "$CHECKING" "$ERROR" "$TRACKER_WARN" "$DOWN" "$STALLED_DL" \
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
