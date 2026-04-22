#!/usr/bin/env bash
# Version: 1.3.3
set -euo pipefail

SCRIPT_VERSION="1.3.3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QBIT_URL="${QBIT_URL:-http://localhost:9003}"
QBIT_USER="${QBIT_USER:-${QBITTORRENTAPI_USERNAME:-admin}}"
QBIT_PASS="${QBIT_PASS:-${QBITTORRENTAPI_PASSWORD:-adminpass}}"
INTERVAL_S=30
ONCE=0
UNTIL_CLEAR=0
ENFORCE_PAUSED_DL=0
ALLOW_FILE=""
EVENTS_JSONL=""
MAX_ITERATIONS=0
DASHBOARD=0
USE_CACHE=1
CACHE_MAX_AGE=30
CACHE_AGENT="${QBIT_CACHE_AGENT:-${SILO_CACHE_AGENT:-/home/michael/dev/tools/silo/bin/silo-cache-agent.py}}"
CACHE_FILE_FALLBACK="${QBIT_CACHE_FALLBACK_FILE:-$HOME/.cache/hashall-qb/torrents-info.json}"
CACHE_CLIENT_ID="$(basename "$0"):$$"
CACHE_PYTHON="${QBIT_CACHE_PYTHON:-}"
declare -a ALLOW_HASHES=()

R=$'\033[0m'
DIM=$'\033[2m'
RED=$'\033[31m'
GREEN=$'\033[32m'
YELLOW=$'\033[33m'
CYAN=$'\033[36m'
BLUE=$'\033[34m'
SHOW_HELP=0
CACHE_FILE_PATH=""
CACHE_META_PATH=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Watches qB torrent state counts (checking/missing/moving/down/up).
Optional watchdog mode pauses unexpected downloading torrents and emits alerts.
Env: QBIT_URL, QBIT_USER, QBIT_PASS

Options:
  --interval N            Poll interval seconds (default: 30)
  --once                  Run one sample then exit
  --until-clear           Exit when checking=0 and moving=0 and down=0
  --enforce-paused-dl     Pause unexpected downloading/stalledDL torrents
  --allow-hash HASH       Allowlist hash that watchdog must not auto-pause (repeatable)
  --allow-file PATH       File with allowlisted hashes (one per line)
  --events-jsonl PATH     Write watchdog events as JSONL
  --max-iterations N      Exit after N polling iterations (default: 0 = infinite)
  --dashboard             Overwrite-in-place dashboard mode (like watch)
  --cache                 Read qB torrents/info via shared cache agent (default)
  --no-cache              Bypass shared cache and query qB API directly
  --cache-max-age N       Max cache age seconds when --cache is enabled (default: 30)
  -h, --help              Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --interval)
    INTERVAL_S="${2:-}"
    shift 2
    ;;
  --once)
    ONCE=1
    shift
    ;;
  --until-clear)
    UNTIL_CLEAR=1
    shift
    ;;
  --enforce-paused-dl)
    ENFORCE_PAUSED_DL=1
    shift
    ;;
  --allow-hash)
    ALLOW_HASHES+=("${2:-}")
    shift 2
    ;;
  --allow-file)
    ALLOW_FILE="${2:-}"
    shift 2
    ;;
  --events-jsonl)
    EVENTS_JSONL="${2:-}"
    shift 2
    ;;
  --max-iterations)
    MAX_ITERATIONS="${2:-}"
    shift 2
    ;;
  --dashboard)
    DASHBOARD=1
    shift
    ;;
  --cache)
    USE_CACHE=1
    shift
    ;;
  --no-cache)
    USE_CACHE=0
    shift
    ;;
  --cache-max-age)
    CACHE_MAX_AGE="${2:-}"
    shift 2
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown argument: $1" >&2
    usage >&2
    exit 2
    ;;
  esac
done

if ! [[ "$INTERVAL_S" =~ ^[0-9]+$ ]] || [[ "$INTERVAL_S" -lt 1 ]]; then
  echo "--interval must be a positive integer" >&2
  exit 2
fi
if ! [[ "$MAX_ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "--max-iterations must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$CACHE_MAX_AGE" =~ ^[0-9]+$ ]] || [[ "$CACHE_MAX_AGE" -lt 0 ]]; then
  echo "--cache-max-age must be a non-negative integer" >&2
  exit 2
fi
if [[ "$USE_CACHE" -eq 1 && ! -f "$CACHE_AGENT" ]]; then
  echo "--cache enabled but cache agent not found: $CACHE_AGENT" >&2
  exit 2
fi

resolve_cache_python() {
  resolve_venv_python() {
    local root="$1" venv_name="" venv_python=""
    [[ -f "$root/.venv_name" ]] || return 1
    venv_name="$(<"$root/.venv_name")"
    [[ -n "$venv_name" ]] || return 1
    venv_python="${HOME}/.venvs/${venv_name}/bin/python3"
    [[ -x "$venv_python" ]] || return 1
    printf '%s\n' "$venv_python"
  }

  if [[ -n "$CACHE_PYTHON" ]]; then
    printf '%s\n' "$CACHE_PYTHON"
    return
  fi

  local agent_realpath="" agent_dir="" repo_root="" hashall_root=""
  agent_realpath="$(readlink -f "$CACHE_AGENT" 2>/dev/null || printf '%s' "$CACHE_AGENT")"
  agent_dir="$(cd "$(dirname "$agent_realpath")" && pwd)"
  repo_root="$(cd "$agent_dir/.." && pwd)"

  if resolve_venv_python "$repo_root" >/dev/null 2>&1; then
    resolve_venv_python "$repo_root"
    return
  fi

  # `silo` exposes thin wrapper scripts that exec the canonical hashall tools.
  if [[ -f "$agent_dir/silo_hashall_shared.py" ]]; then
    hashall_root="${HASHALL_ROOT:-/home/michael/dev/work/hashall}"
    if resolve_venv_python "$hashall_root" >/dev/null 2>&1; then
      resolve_venv_python "$hashall_root"
      return
    fi
  fi

  command -v python3
}

CACHE_PYTHON="$(resolve_cache_python)"

load_cache_file_fallback() {
  local max_age="$1"
  local now epoch age raw
  [[ -f "$CACHE_FILE_FALLBACK" ]] || return 1
  epoch="$(stat -c '%Y' "$CACHE_FILE_FALLBACK" 2>/dev/null || true)"
  [[ -n "$epoch" && "$epoch" =~ ^[0-9]+$ ]] || return 1
  now="$(date +%s)"
  age=$((now - epoch))
  (( age >= 0 )) || age=0
  if (( max_age > 0 && age > max_age )); then
    return 1
  fi
  raw="$(<"$CACHE_FILE_FALLBACK")"
  jq -e . >/dev/null 2>&1 <<<"$raw" || return 1
  printf '%s' "$raw"
}

COOKIE_FILE="$(mktemp)"
_on_exit() {
  rm -f "$COOKIE_FILE"
  if [[ "$DASHBOARD" -eq 1 ]]; then printf '\n'; fi
}
trap '_on_exit' EXIT
if [[ "$ENFORCE_PAUSED_DL" -eq 1 && -n "$EVENTS_JSONL" ]]; then
  mkdir -p "$(dirname "$EVENTS_JSONL")"
fi

declare -A ALLOW_HASH_MAP=()
for h in "${ALLOW_HASHES[@]}"; do
  key="$(tr '[:upper:]' '[:lower:]' <<<"${h//[[:space:]]/}")"
  [[ -n "$key" ]] && ALLOW_HASH_MAP["$key"]=1
done
if [[ -n "$ALLOW_FILE" && -f "$ALLOW_FILE" ]]; then
  while IFS= read -r line; do
    line="${line%%#*}"
    key="$(tr '[:upper:]' '[:lower:]' <<<"${line//[[:space:]]/}")"
    [[ -n "$key" ]] && ALLOW_HASH_MAP["$key"]=1
  done <"$ALLOW_FILE"
fi

if [[ "$ENFORCE_PAUSED_DL" -eq 1 && -z "$EVENTS_JSONL" ]]; then
  stamp="$(TZ=America/New_York date +%Y%m%d-%H%M%S)"
  EVENTS_JSONL="$HOME/.logs/hashall/reports/rehome-normalize/qb-paused-dl-watchdog-${stamp}.jsonl"
  mkdir -p "$(dirname "$EVENTS_JSONL")"
fi

if [[ "$DASHBOARD" -eq 0 ]]; then
  echo "watchdog_config interval_s=${INTERVAL_S} once=${ONCE} until_clear=${UNTIL_CLEAR} enforce_paused_dl=${ENFORCE_PAUSED_DL} allow_count=${#ALLOW_HASH_MAP[@]} events_jsonl=${EVENTS_JSONL:-none} max_iterations=${MAX_ITERATIONS} cache=${USE_CACHE} cache_max_age=${CACHE_MAX_AGE}"
fi

api_login() {
  curl -sS -c "$COOKIE_FILE" \
    --data-urlencode "username=${QBIT_USER}" \
    --data-urlencode "password=${QBIT_PASS}" \
    "${QBIT_URL}/api/v2/auth/login" >/dev/null
}

api_post_status() {
  local endpoint="$1"
  local hashes="$2"
  curl -sS -o /dev/null -w "%{http_code}" \
    -b "$COOKIE_FILE" \
    --data-urlencode "hashes=${hashes}" \
    "${QBIT_URL}${endpoint}" || echo "000"
}

pause_with_fallback() {
  local hashes="$1"
  local code=""

  if ! api_login >/dev/null 2>&1; then
    PAUSE_ACTION_RESULT="pause_failed_login"
    return 1
  fi

  code="$(api_post_status "/api/v2/torrents/pause" "$hashes")"
  case "$code" in
  200 | 202)
    PAUSE_ACTION_RESULT="paused"
    return 0
    ;;
  404)
    code="$(api_post_status "/api/v2/torrents/stop" "$hashes")"
    if [[ "$code" == "200" || "$code" == "202" ]]; then
      PAUSE_ACTION_RESULT="paused_via_stop"
      return 0
    fi
    PAUSE_ACTION_RESULT="pause_failed_stop_http_${code}"
    return 1
    ;;
  *)
    PAUSE_ACTION_RESULT="pause_failed_http_${code}"
    return 1
    ;;
  esac
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
print('good' if age <= 10 else 'watch' if age <= 30 else 'bad')
PY
}

cache_status_text() {
  local detail="$1" age="$2"
  local status
  status="${detail:-fresh}"
  printf '%s·%ss' "$status" "$age"
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

reset_qb_cache() {
  local cf="${CACHE_FILE_PATH:-$CACHE_FILE_FALLBACK}"
  local mf="${CACHE_META_PATH:-${cf%.json}.meta.json}"
  rm -f "$cf" "$mf"
  QBIT_URL="$QBIT_URL" QBIT_USER="$QBIT_USER" QBIT_PASS="$QBIT_PASS" \
    "$CACHE_PYTHON" "$CACHE_AGENT" --ensure-daemon --max-age 0 >/dev/null 2>&1 || true
}

print_dashboard() {
  local ts="$1"
  local transport_label="$2"
  local transport_detail="$3"
  local cache_age_s="$4"
  local active_leases="$5"
  local checking="$6"
  local error_count="$7"
  local missing="$8"
  local moving="$9"
  local down="${10}"
  local stalled_dl="${11}"
  local stopped_dl="${12}"
  local stalled_up="${13}"
  local uploading="${14}"
  local stopped_up="${15}"
  local queued_up="${16}"
  local unexpected_down="${17}"
  local total="${18}"
  local show_unexpected="${19}"
  local cache_dot cache_age_color ts_short title footer_label cache_status
  local cache_line1 cache_line2 attn_total dl_total ul_total
  local -a lines=() bars=()
  local width=0 vis

  cache_dot="${GREEN}●${R}"
  [[ -n "$transport_detail" ]] && cache_dot="${YELLOW}●${R}"
  [[ "$transport_detail" == *dmn↓* || "$transport_detail" == *unavailable* ]] && cache_dot="${RED}●${R}"
  cache_age_color="$(cache_age_sev "$cache_age_s")"
  ts_short="$(date -d "$ts" '+%m-%d-%y %H:%M' 2>/dev/null || printf '%s' "$ts")"
  attn_total=$((error_count + missing + moving + unexpected_down))
  dl_total=$((down + stopped_dl))
  ul_total=$((uploading + stalled_up + stopped_up + queued_up))
  title="qBittorrent v${SCRIPT_VERSION}"
  if [[ "$transport_label" == "Direct" ]]; then
    cache_line1="Direct: ${cache_dot} $(color_num "${INTERVAL_S}s" info) ${DIM}·${R} api"
    cache_line2="live"
  else
    cache_line1="Cache: ${cache_dot} $(color_num "${INTERVAL_S}s" info) ${DIM}·${R} ttl$(color_num "${CACHE_MAX_AGE}s" info)"
    cache_status="$(cache_status_text "${transport_detail:-fresh}" "${cache_age_s:-0}")"
    cache_line2="$(color_num "$cache_status" "$cache_age_color")"
  fi

  lines+=("$cache_line1" "$cache_line2")
  lines+=("$(printf '%-9s' "checking") : $(printf '%5s' "$(color_num "$checking" watch)")")
  lines+=("$(printf '%-9s' "error") : $(printf '%5s' "$(color_num "$error_count" bad)")")
  lines+=("$(printf '%-9s' "missing") : $(printf '%5s' "$(color_num "$missing" bad)")")
  lines+=("$(printf '%-9s' "moving") : $(printf '%5s' "$(color_num "$moving" watch)")")
  if [[ "$show_unexpected" -eq 1 ]]; then
    lines+=("$(printf '%-9s' "unexpected") : $(printf '%5s' "$(color_num "$unexpected_down" bad)")")
  fi
  lines+=("$(printf '%-9s' "active") : $(printf '%5s' "$(color_num "$down" good)")")
  lines+=("$(printf '%-9s' "stalled") : $(printf '%5s' "$(color_num "$stalled_dl" watch)")")
  lines+=("$(printf '%-9s' "stopped") : $(printf '%5s' "$(color_num "$stopped_dl" watch)")")
  lines+=("$(printf '%-9s' "active") : $(printf '%5s' "$(color_num "$uploading" good)")")
  lines+=("$(printf '%-9s' "idle") : $(printf '%5s' "$(color_num "$stalled_up" good)")")
  lines+=("$(printf '%-9s' "queued") : $(printf '%5s' "$(color_num "$queued_up" watch)")")
  lines+=("$(printf '%-9s' "stopped") : $(printf '%5s' "$(color_num "$stopped_up" dim)")")
  bars+=("Attention:$(color_num "$attn_total" $([[ "$attn_total" -gt 0 ]] && echo watch || echo dim))")
  bars+=("DL:$(color_num "$dl_total" $([[ "$dl_total" -gt 0 ]] && echo watch || echo dim))")
  bars+=("UL:$(color_num "$ul_total" good)")
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
  local idx=2
  box_line "${lines[$idx]}" "$width"; ((idx++))
  box_line "${lines[$idx]}" "$width"; ((idx++))
  box_line "${lines[$idx]}" "$width"; ((idx++))
  box_line "${lines[$idx]}" "$width"; ((idx++))
  if [[ "$show_unexpected" -eq 1 ]]; then
    box_line "${lines[$idx]}" "$width"; ((idx++))
  fi
  box_bar "${bars[1]}" "$width"
  box_line "${lines[$idx]}" "$width"; ((idx++))
  box_line "${lines[$idx]}" "$width"; ((idx++))
  box_line "${lines[$idx]}" "$width"; ((idx++))
  box_bar "${bars[2]}" "$width"
  box_line "${lines[$idx]}" "$width"; ((idx++))
  box_line "${lines[$idx]}" "$width"; ((idx++))
  box_line "${lines[$idx]}" "$width"; ((idx++))
  box_line "${lines[$idx]}" "$width"
  box_bar "${bars[3]}" "$width"
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
    c) USE_CACHE=1; _FORCE_REDRAW=1 ;;
    d) USE_CACHE=0; _FORCE_REDRAW=1 ;;
    r) reset_qb_cache; USE_CACHE=1; _FORCE_REDRAW=1 ;;
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
  TORRENTS_JSON="[]"
  CACHE_TRANSPORT_LABEL="Direct"
  CACHE_TRANSPORT_DETAIL=""
  CACHE_AGE_S=""
  CACHE_ACTIVE_LEASES="0"
  if [[ "$USE_CACHE" -eq 1 ]]; then
    _status=""
    if _status="$(QBIT_URL="$QBIT_URL" QBIT_USER="$QBIT_USER" QBIT_PASS="$QBIT_PASS" \
      "$CACHE_PYTHON" "$CACHE_AGENT" --status 2>/dev/null)" && jq -e . >/dev/null 2>&1 <<<"$_status"; then
      CACHE_TRANSPORT_LABEL="Cache"
      CACHE_AGE_S="$(jq -r 'if .cache_age_s == null then "" else ((.cache_age_s | tonumber) * 10 | floor / 10 | tostring) end' <<<"$_status")"
      CACHE_ACTIVE_LEASES="$(jq -r '.active_lease_count // 0' <<<"$_status")"
      CACHE_FILE_PATH="$(jq -r '.cache_file // empty' <<<"$_status")"
      CACHE_META_PATH="$(jq -r '.meta_file // empty' <<<"$_status")"
      CACHE_TRANSPORT_DETAIL="$(jq -r '
        (if .daemon_running == true then "" else "dmn↓" end)
      ' <<<"$_status")"
      [[ "$CACHE_TRANSPORT_DETAIL" == "null" ]] && CACHE_TRANSPORT_DETAIL=""
    fi
    _raw=""
    if ! _raw="$(QBIT_URL="$QBIT_URL" QBIT_USER="$QBIT_USER" QBIT_PASS="$QBIT_PASS" \
      "$CACHE_PYTHON" "$CACHE_AGENT" \
        --max-age "$CACHE_MAX_AGE" \
        --requested-interval "$INTERVAL_S" \
        --client-id "$CACHE_CLIENT_ID" \
        --ensure-daemon \
        2>/dev/null)"; then
      if _raw="$(load_cache_file_fallback "$CACHE_MAX_AGE" 2>/dev/null)"; then
        TORRENTS_JSON="$_raw"
      else
        FETCH_ERROR="cache_fetch_failed"
      fi
    elif ! jq -e . >/dev/null 2>&1 <<<"$_raw"; then
      if _raw="$(load_cache_file_fallback "$CACHE_MAX_AGE" 2>/dev/null)"; then
        TORRENTS_JSON="$_raw"
      else
        FETCH_ERROR="cache_invalid_json"
      fi
    else
      TORRENTS_JSON="$_raw"
    fi
  else
    if ! api_login >/dev/null 2>&1; then
      FETCH_ERROR="login_failed"
    fi

    if [[ -z "$FETCH_ERROR" ]]; then
      _raw=""
      if ! _raw="$(curl -sS -b "$COOKIE_FILE" "${QBIT_URL}/api/v2/torrents/info" 2>&1)"; then
        FETCH_ERROR="fetch_failed"
      elif ! jq -e . >/dev/null 2>&1 <<<"$_raw"; then
        FETCH_ERROR="invalid_json"
      else
        TORRENTS_JSON="$_raw"
      fi
    fi
  fi

  if [[ -n "$FETCH_ERROR" ]]; then
    _err_ts="$(date '+%F %T')"
    if [[ "$DASHBOARD" -eq 1 ]]; then
      printf '\033[2J\033[H'
      printf '── qBittorrent v%s ── ERROR\n' "$SCRIPT_VERSION"
      if [[ "$USE_CACHE" -eq 1 ]]; then
        printf 'Cache: ● %ss%s\n' "$INTERVAL_S" "${CACHE_AGE_S:+  age ${CACHE_AGE_S}s}"
      else
        printf 'Direct: ● %ss\n' "$INTERVAL_S"
      fi
      printf '%s\n' "$_err_ts"
      printf '────────────────────────\n'
      printf '%-12s : %s\n' "error" "$FETCH_ERROR"
      printf '%-12s : %5s\n' "total" "-"
      _FORCE_REDRAW=0
    else
      printf '%s error fetch_error=%s\n' "$_err_ts" "$FETCH_ERROR"
    fi
    if [[ "$ONCE" -eq 1 ]]; then
      exit 1
    fi
    sleep "$INTERVAL_S" &
    _SLEEP_PID=$!
    wait "$_SLEEP_PID" 2>/dev/null || true
    kill "$_SLEEP_PID" 2>/dev/null || true
    continue
  fi

  read -r CHECKING ERROR_COUNT MISSING MOVING DOWN STALLED_DL UP COUNT_ZERO COUNT_PARTIAL STOPPED_UP STOPPED_DL STALLED_UP UPLOADING QUEUED_UP TOTAL TOP_STATES <<<"$(jq -r '
    [
      ([.[] | (.state // "" | ascii_downcase) | select(startswith("checking"))] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "error")] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "missingfiles")] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "moving")] | length),
      ([.[] | select(
        (.state // "" | ascii_downcase) == "downloading"
        or (.state // "" | ascii_downcase) == "stalleddl"
        or (.state // "" | ascii_downcase) == "queueddl"
        or (.state // "" | ascii_downcase) == "forceddl"
        or (.state // "" | ascii_downcase) == "metadl"
      )] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "stalleddl")] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "uploading" or (.state // "" | ascii_downcase) == "stalledup")] | length),
      ([.[] | select((.progress // 0) <= 0.0)] | length),
      ([.[] | select((.progress // 0) > 0.0 and (.progress // 0) < 1.0)] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "stoppedup")] | length),
      ([.[] | select(
        (.state // "" | ascii_downcase) == "stoppeddl"
        or (.state // "" | ascii_downcase) == "pauseddl"
      )] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "stalledup")] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "uploading")] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "queuedup")] | length),
      (length),
      (
        group_by(.state // "UNKNOWN")
        | map({s: (.[0].state // "UNKNOWN"), c: length})
        | sort_by(-.c)
        | .[:8]
        | map("\(.s)=\(.c)")
        | join(",")
      )
    ] | @tsv
  ' <<<"$TORRENTS_JSON")"

  mapfile -t DOWN_HASHES_RAW < <(jq -r '
    .[]
    | select(
        (.state // "" | ascii_downcase) == "downloading"
        or (.state // "" | ascii_downcase) == "stalleddl"
        or (.state // "" | ascii_downcase) == "queueddl"
        or (.state // "" | ascii_downcase) == "forceddl"
        or (.state // "" | ascii_downcase) == "metadl"
      )
    | (.hash // "" | ascii_downcase)
  ' <<<"$TORRENTS_JSON")
  declare -a UNEXPECTED_DOWN=()
  declare -A SEEN_HASH=()
  for torrent_hash in "${DOWN_HASHES_RAW[@]}"; do
    [[ -z "$torrent_hash" ]] && continue
    if [[ -n "${SEEN_HASH[$torrent_hash]+x}" ]]; then
      continue
    fi
    SEEN_HASH["$torrent_hash"]=1
    if [[ -z "${ALLOW_HASH_MAP[$torrent_hash]+x}" ]]; then
      UNEXPECTED_DOWN+=("$torrent_hash")
    fi
  done

  paused_now=0
  if [[ "$ENFORCE_PAUSED_DL" -eq 1 && "${#UNEXPECTED_DOWN[@]}" -gt 0 ]]; then
    pause_hashes="$(
      IFS='|'
      echo "${UNEXPECTED_DOWN[*]}"
    )"
    pause_action="pause_failed"
    PAUSE_ACTION_RESULT="pause_failed"
    if pause_with_fallback "$pause_hashes"; then
      pause_action="$PAUSE_ACTION_RESULT"
      paused_now="${#UNEXPECTED_DOWN[@]}"
    else
      pause_action="$PAUSE_ACTION_RESULT"
    fi
    alert_hashes="$(
      IFS=,
      echo "${UNEXPECTED_DOWN[*]}"
    )"
    if [[ "$DASHBOARD" -eq 0 ]]; then
      printf '%s ALERT unexpected_downloading action=%s count=%s hashes=%s\n' \
        "$(date '+%F %T')" "$pause_action" "${#UNEXPECTED_DOWN[@]}" "$alert_hashes"
    fi
    if [[ -n "$EVENTS_JSONL" ]]; then
      hashes_json="$(printf '%s\n' "${UNEXPECTED_DOWN[@]}" | jq -Rsc 'split("\n")[:-1]')"
      jq -cn \
        --arg ts "$(date '+%F %T')" \
        --arg action "$pause_action" \
        --argjson count "${#UNEXPECTED_DOWN[@]}" \
        --argjson paused "$paused_now" \
        --argjson hashes "$hashes_json" \
        '{ts:$ts,event:"unexpected_downloading",action:$action,count:$count,paused:$paused,hashes:$hashes}' \
        >>"$EVENTS_JSONL"
    fi
  fi

  if [[ "$DASHBOARD" -eq 1 ]]; then
    printf '\033[2J\033[H'
    _FORCE_REDRAW=0
    print_dashboard \
      "$(date '+%F %T')" \
      "$CACHE_TRANSPORT_LABEL" "$CACHE_TRANSPORT_DETAIL" "$CACHE_AGE_S" "$CACHE_ACTIVE_LEASES" \
      "$CHECKING" "$ERROR_COUNT" "$MISSING" "$MOVING" "$DOWN" "$STALLED_DL" \
      "$STOPPED_DL" "$STALLED_UP" "$UPLOADING" "$STOPPED_UP" "$QUEUED_UP" \
      "${#UNEXPECTED_DOWN[@]}" "$TOTAL" "$ENFORCE_PAUSED_DL"
  else
    printf '%s checking=%s missing=%s moving=%s down=%s up=%s unexpected_down=%s paused_now=%s count_zero=%s count_partial=%s top=%s stoppedUP=%s stoppedDL=%s stalledUP=%s uploading=%s queuedUP=%s\n' \
      "$(date '+%F %T')" "$CHECKING" "$MISSING" "$MOVING" "$DOWN" "$UP" "${#UNEXPECTED_DOWN[@]}" "$paused_now" "$COUNT_ZERO" "$COUNT_PARTIAL" "$TOP_STATES" "$STOPPED_UP" "$STOPPED_DL" "$STALLED_UP" "$UPLOADING" "$QUEUED_UP"
  fi
  if [[ "$UNTIL_CLEAR" -eq 1 && "$CHECKING" -eq 0 && "$MOVING" -eq 0 && "$DOWN" -eq 0 ]]; then
    if [[ "$DASHBOARD" -eq 0 ]]; then
      echo "done checking=0 moving=0 down=0"
    fi
    exit 0
  fi

  if [[ "$ONCE" -eq 1 ]]; then
    exit 0
  fi
  if [[ "$MAX_ITERATIONS" -gt 0 && "$iteration" -ge "$MAX_ITERATIONS" ]]; then
    if [[ "$DASHBOARD" -eq 0 ]]; then
      echo "done max_iterations=${MAX_ITERATIONS}"
    fi
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
