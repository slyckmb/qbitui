#!/usr/bin/env bash
# Version: 1.4.1
# rt-status.sh — rTorrent state dashboard
#
# Reads rTorrent state from the shared silo cache by default.
# Direct XMLRPC access is available only via --direct for one-off diagnostics.
set -euo pipefail

SCRIPT_VERSION="1.4.1"
RT_CONTAINER="${RT_CONTAINER:-rtorrent_vpn}"
RT_RPC_URL="${RT_RPC_URL:-http://localhost:8000/}"
RT_CACHE_SUMMARY="${RT_CACHE_SUMMARY:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/rt-cache-summary.py}"
RT_CACHE_DAEMON="${RT_CACHE_DAEMON:-${SILO_RT_DAEMON_SCRIPT:-/home/michael/dev/tools/silo/bin/silo-rt-cache-daemon.py}}"
RT_CACHE_PYTHON="${RT_CACHE_PYTHON:-python3}"
RT_CACHE_MAX_AGE="${RT_CACHE_MAX_AGE:-60}"
INTERVAL_S=30
ONCE=0
UNTIL_CLEAR=0
MAX_ITERATIONS=0
DASHBOARD=0
DIRECT_MODE=0

R=$'\033[0m'
DIM=$'\033[2m'
BOLD=$'\033[1m'
RED=$'\033[31m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
BLUE=$'\033[34m'
CYAN=$'\033[36m'
SHOW_HELP=0

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
    '<param><value><string>d.is_active=</string></value></param>'
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

def derive_state(state, is_active, hashing, complete, down_rate, up_rate, message):
    if hashing > 0:
        return 'checking'
    if state == 0:
        return 'stoppedUP' if complete == 1 else 'stoppedDL'
    # state == 1 (started in rtorrent) but is_active == 0 means paused
    if is_active == 0:
        return 'pausedUP' if complete == 1 else 'pausedDL'
    # state == 1 (started/active)
    if complete == 0:
        # incomplete + message = download is stuck/broken (fatal)
        return 'error' if (message and message.strip()) else ('downloading' if down_rate > 0 else 'stalledDL')
    # complete == 1: item is seeding regardless of tracker announce result
    return 'uploading' if up_rate > 0 else 'stalledUP'

raw = parse_multicall(xml)
output = []
for fields in raw:
    if len(fields) < 11:
        continue
    hash_     = str(fields[0])
    state     = int(fields[1]) if isinstance(fields[1], int) else 0
    is_active = int(fields[2]) if isinstance(fields[2], int) else 0
    hashing   = int(fields[3]) if isinstance(fields[3], int) else 0
    complete  = int(fields[4]) if isinstance(fields[4], int) else 0
    down_rate = int(fields[5]) if isinstance(fields[5], int) else 0
    up_rate   = int(fields[6]) if isinstance(fields[6], int) else 0
    message   = str(fields[7]) if fields[7] is not None else ''
    label     = str(fields[8]) if fields[8] is not None else ''
    size      = int(fields[9]) if isinstance(fields[9], int) else 0
    done      = int(fields[10]) if isinstance(fields[10], int) else 0
    progress  = (done / size) if size > 0 else (1.0 if complete else 0.0)

    output.append({
        'hash':      hash_.lower(),
        'state':     derive_state(state, is_active, hashing, complete, down_rate, up_rate, message),
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

strip_ansi() {
  sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g' <<<"$1"
}

visible_width() {
  local raw
  raw="$(strip_ansi "$1")"
  printf '%s' "${#raw}"
}

repeat_char() {
  local char="$1" count="$2"
  (( count <= 0 )) && return 0
  local out="" i
  for ((i = 0; i < count; i++)); do
    out+="$char"
  done
  printf '%s' "$out"
}

color_num() {
  local value="$1" sev="$2"
  case "$sev" in
    good) printf '%s%s%s' "$GREEN" "$value" "$R" ;;
    watch) printf '%s%s%s' "$YELLOW" "$value" "$R" ;;
    warn) printf '%s%s%s' "$BLUE" "$value" "$R" ;;
    bad) printf '%s%s%s' "$RED" "$value" "$R" ;;
    info) printf '%s%s%s' "$CYAN" "$value" "$R" ;;
    dim) printf '%s%s%s' "$DIM" "$value" "$R" ;;
    *) printf '%s' "$value" ;;
  esac
}

cache_age_sev() {
  local age="${1:-}"
  [[ -z "$age" ]] && { printf 'dim'; return; }
  python3 - <<'PY' "$age"
import sys
age=float(sys.argv[1])
print('good' if age <= 10 else 'watch' if age <= 60 else 'bad')
PY
}

cache_status_text() {
  local detail="$1" age="$2"
  local freshness suffix
  freshness="fresh"
  [[ "$detail" == *stale* ]] && freshness="stale"
  [[ "$detail" == *missing* ]] && freshness="missing"
  suffix="${detail//stale/}"
  suffix="${suffix//missing/}"
  suffix="${suffix//fresh/}"
  suffix="${suffix#·}"
  suffix="${suffix%·}"
  if [[ -n "$suffix" ]]; then
    printf '%s:%s·%ss' "$freshness" "$suffix" "$age"
  else
    printf '%s·%ss' "$freshness" "$age"
  fi
}

box_top() {
  local label="$1" width="$2"
  local core=" ${label} "
  local pad
  pad=$((width - $(visible_width "$core")))
  (( pad < 1 )) && pad=1
  printf '┌%s%s┐\n' "$core" "$(repeat_char '─' "$pad")"
}

box_line() {
  local text="$1" width="$2"
  local vis pad
  vis="$(visible_width "$text")"
  pad=$((width - vis))
  (( pad < 0 )) && pad=0
  printf '│%s%s│\n' "$text" "$(repeat_char ' ' "$pad")"
}

box_bar() {
  local label="$1" width="$2"
  local core="┤${label}├"
  local core_w left right
  core_w="$(visible_width "$core")"
  left=$(((width - core_w) / 2))
  right=$((width - core_w - left))
  printf '├%s%s%s┤\n' "$(repeat_char '─' "$left")" "$core" "$(repeat_char '─' "$right")"
}

box_footer() {
  local label="$1" width="$2"
  local core="┤${label}├"
  local fill left right
  fill=$((width - $(visible_width "$core")))
  left=$((fill / 2))
  right=$((fill - left))
  printf '└%s%s%s┘\n' "$(repeat_char '─' "$left")" "$core" "$(repeat_char '─' "$right")"
}

reset_rt_cache() {
  rm -f "$HOME/.cache/silo-rt/torrents.json" "$HOME/.cache/silo-rt/torrents.meta.json"
  timeout 15s "$RT_CACHE_PYTHON" "$RT_CACHE_DAEMON" --once >/dev/null 2>&1 || true
}

print_dashboard() {
  local ts="$1"
  local transport_label="$2"
  local transport_detail="$3"
  local cache_age_s="$4"
  local active_leases="$5"
  local checking="$6"
  local error="$7"
  local trk_warn="$8"
  local trk_breakdown="$9"
  local down="${10}"
  local stalled_dl="${11}"
  local stalled_up="${12}"
  local uploading="${13}"
  local stopped_up="${14}"
  local stopped_dl="${15}"
  local total="${16}"
  local cache_dot cache_age_color transport_extra ts_short title footer_label
  local cache_line1 cache_line2
  local attn_total dl_total ul_total
  local -a lines=() bars=()
  local width=0 vis
  cache_dot="${GREEN}●${R}"
  [[ "$transport_detail" == *stale* ]] && cache_dot="${YELLOW}●${R}"
  [[ "$transport_detail" == *dmn↓* || "$transport_detail" == *missing* ]] && cache_dot="${RED}●${R}"
  cache_age_color="$(cache_age_sev "$cache_age_s")"
  transport_extra="$(cache_status_text "${transport_detail:-fresh}" "${cache_age_s:-0}")"
  ts_short="$(date -d "$ts" '+%m-%d-%y %H:%M' 2>/dev/null || printf '%s' "$ts")"
  attn_total=$((checking + error))
  dl_total=$((down + stalled_dl + stopped_dl))
  ul_total=$((uploading + stalled_up + stopped_up))
  title="rTorrent v${SCRIPT_VERSION}"
  if [[ "$transport_label" == "direct" ]]; then
    cache_line1="Direct: ${cache_dot} $(color_num "${INTERVAL_S}s" info) ${DIM}·${R} xmlrpc"
    cache_line2="live"
  else
    cache_line1="Cache: ${cache_dot} $(color_num "${INTERVAL_S}s" info) ${DIM}·${R} ttl$(color_num "${RT_CACHE_MAX_AGE}s" info)"
    cache_line2="$(color_num "$transport_extra" "$cache_age_color")"
  fi

  lines+=("$cache_line1" "$cache_line2")
  lines+=("$(printf '%-9s' "checking") : $(printf '%5s' "$(color_num "$checking" watch)")")
  lines+=("$(printf '%-9s' "fatal") : $(printf '%5s' "$(color_num "$error" bad)")")
  lines+=("$(printf '%-9s' "active") : $(printf '%5s' "$(color_num "$down" good)")")
  lines+=("$(printf '%-9s' "stalled") : $(printf '%5s' "$(color_num "$stalled_dl" watch)")")
  lines+=("$(printf '%-9s' "stopped") : $(printf '%5s' "$(color_num "$stopped_dl" dim)")")
  lines+=("$(printf '%-9s' "active") : $(printf '%5s' "$(color_num "$uploading" good)")")
  lines+=("$(printf '%-9s' "idle") : $(printf '%5s' "$(color_num "$stalled_up" good)")")
  lines+=("$(printf '%-9s' "stopped") : $(printf '%5s' "$(color_num "$stopped_up" dim)")")
  bars+=("Attention:$(color_num "$attn_total" warn)")
  bars+=("DL:$(color_num "$dl_total" $([[ "$dl_total" -gt 0 ]] && echo watch || echo dim))")
  bars+=("UL:$(color_num "$ul_total" good)")
  if [[ "$trk_warn" -gt 0 ]] 2>/dev/null; then
    bars+=("trk_warn:$(color_num "$trk_warn" warn)")
    if [[ -n "$trk_breakdown" ]]; then
      while IFS=$'\t' read -r cat cnt; do
        [[ -z "$cat" ]] && continue
        lines+=("$(printf '%-9s' "$cat") : $(printf '%5s' "$(color_num "$cnt" $([[ "$cat" == auth_err || "$cat" == deleted ]] && echo bad || echo watch))")")
      done <<<"$trk_breakdown"
    fi
  fi
  bars+=("total:$(color_num "$total" info)")

  for line in "$title" "${lines[@]}" "${bars[@]}" "$ts_short"; do
    vis="$(visible_width "$line")"
    (( vis > width )) && width="$vis"
  done
  if [[ "$SHOW_HELP" -eq 1 ]]; then
    vis="$(visible_width 'q quit  c cache  d direct  r reset  +/- interval  ? help')"
    (( vis > width )) && width="$vis"
  fi

  box_top "$title" "$width"
  box_line "$cache_line1" "$width"
  box_line "$cache_line2" "$width"
  box_bar "${bars[0]}" "$width"
  box_line "${lines[2]}" "$width"
  box_line "${lines[3]}" "$width"
  box_bar "${bars[1]}" "$width"
  box_line "${lines[4]}" "$width"
  box_line "${lines[5]}" "$width"
  box_line "${lines[6]}" "$width"
  box_bar "${bars[2]}" "$width"
  box_line "${lines[7]}" "$width"
  box_line "${lines[8]}" "$width"
  box_line "${lines[9]}" "$width"
  if [[ "$trk_warn" -gt 0 ]] 2>/dev/null; then
    box_bar "${bars[3]}" "$width"
    local idx
    for ((idx = 10; idx < ${#lines[@]}; idx++)); do
      box_line "${lines[$idx]}" "$width"
    done
    box_bar "${bars[4]}" "$width"
  else
    box_bar "${bars[3]}" "$width"
  fi
  footer_label="$ts_short"
  [[ "$SHOW_HELP" -eq 1 ]] && footer_label='q quit  c cache  d direct  r reset  +/- interval  ? help'
  box_footer "$footer_label" "$width"
}

_FORCE_REDRAW=0
_on_winch() { _FORCE_REDRAW=1; }
[[ "$DASHBOARD" -eq 1 ]] && trap '_on_winch' SIGWINCH

handle_dashboard_key() {
  local key="$1"
  case "$key" in
    q) exit 0 ;;
    c) DIRECT_MODE=0; _FORCE_REDRAW=1 ;;
    d) DIRECT_MODE=1; _FORCE_REDRAW=1 ;;
    r) reset_rt_cache; DIRECT_MODE=0; _FORCE_REDRAW=1 ;;
    '+') ((INTERVAL_S > 1)) && INTERVAL_S=$((INTERVAL_S - 1)); _FORCE_REDRAW=1 ;;
    '-') INTERVAL_S=$((INTERVAL_S + 1)); _FORCE_REDRAW=1 ;;
    '?') SHOW_HELP=$((1 - SHOW_HELP)); _FORCE_REDRAW=1 ;;
  esac
}

dashboard_pause() {
  local deadline key
  deadline=$((SECONDS + INTERVAL_S))
  while (( SECONDS < deadline )); do
    if read -rsn1 -t 0.2 key; then
      handle_dashboard_key "$key"
      return 0
    fi
  done
}

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
      RT_CACHE_AGE_S=""
      RT_ACTIVE_LEASES="0"
      read -r CHECKING ERROR DOWN STALLED_DL PAUSED_DL PAUSED_UP SEEDING_UP STALLED_UP UPLOADING STOPPED_UP STOPPED_DL TOTAL TOP_STATES <<<"$(jq -r '
        [
          ([.[] | select(.state == "checking")]                             | length),
          ([.[] | select(.state == "error")]                                | length),
          ([.[] | select(.state == "downloading")]                          | length),
          ([.[] | select(.state == "stalledDL")]                            | length),
          ([.[] | select(.state == "pausedDL")]                             | length),
          ([.[] | select(.state == "pausedUP")]                             | length),
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
        (if .freshness != "fresh" then (.freshness // "?") else "" end) +
        (if .daemon_running == false then "·dmn↓" else "" end)
      ' <<<"$SUMMARY_JSON")"
      RT_TRANSPORT_DETAIL="${RT_TRANSPORT_DETAIL#·}"
      RT_CACHE_AGE_S="$(jq -r 'if .cache_age_s == null then "" else ((.cache_age_s | tonumber) * 10 | floor / 10 | tostring) end' <<<"$SUMMARY_JSON")"
      RT_ACTIVE_LEASES="$(jq -r '.active_leases // 0' <<<"$SUMMARY_JSON")"
      if [[ "$(jq -r '.ok' <<<"$SUMMARY_JSON")" != "true" ]]; then
        FETCH_ERROR="$(jq -r '.last_error // .status_error // "cache_missing"' <<<"$SUMMARY_JSON")"
      else
        read -r CHECKING ERROR DOWN STALLED_DL PAUSED_DL PAUSED_UP STALLED_UP UPLOADING STOPPED_UP STOPPED_DL TOTAL TRACKER_WARN TOP_STATES <<<"$(jq -r '
          [
            ((.states.checkingDL // 0) + (.states.checkingUP // 0) + (.states.checking // 0)),
            (.error_fatal // .states.error_fatal // .states.error // 0),
            (.states.downloading // 0),
            (.states.stalledDL // 0),
            (.states.pausedDL // 0),
            (.states.pausedUP // 0),
            (.states.stalledUP // 0),
            (.states.uploading // 0),
            (.states.stoppedUP // 0),
            (.states.stoppedDL // 0),
            (.items // 0),
            (.tracker_warn_total // 0),
            ((.top_states // []) | join(","))
          ] | @tsv
        ' <<<"$SUMMARY_JSON")"
        TRACKER_WARN_BREAKDOWN="$(jq -r '
          (.tracker_warn_by_kind // {})
          | to_entries
          | sort_by(-.value)[]
          | "\(.key)\t\(.value)"
        ' <<<"$SUMMARY_JSON" 2>/dev/null)" || true
        SEEDING_UP=$((STALLED_UP + UPLOADING))
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
      "$RT_TRANSPORT_LABEL" "$RT_TRANSPORT_DETAIL" "$RT_CACHE_AGE_S" "$RT_ACTIVE_LEASES" \
      "$CHECKING" "$ERROR" "$TRACKER_WARN" "$TRACKER_WARN_BREAKDOWN" \
      "$DOWN" "$STALLED_DL" "$STALLED_UP" "$UPLOADING" \
      "$STOPPED_UP" "$STOPPED_DL" "$TOTAL"
  else
    printf '%s transport=%s detail=%q checking=%s error=%s trk_warn=%s downloading=%s stalledDL=%s pausedDL=%s pausedUP=%s seeding=%s stalledUP=%s uploading=%s stoppedUP=%s stoppedDL=%s total=%s top=%s\n' \
      "$(date '+%F %T')" \
      "$RT_TRANSPORT_LABEL" "$RT_TRANSPORT_DETAIL" \
      "$CHECKING" "$ERROR" "$TRACKER_WARN" "$DOWN" "$STALLED_DL" "$PAUSED_DL" "$PAUSED_UP" \
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

  if [[ "$DASHBOARD" -eq 1 ]]; then
    dashboard_pause
  else
    sleep "$INTERVAL_S" &
    _SLEEP_PID=$!
    wait "$_SLEEP_PID" 2>/dev/null || true
    kill "$_SLEEP_PID" 2>/dev/null || true
  fi
done
