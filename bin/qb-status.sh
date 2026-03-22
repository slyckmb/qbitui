#!/usr/bin/env bash
# Version: 1.2.3
set -euo pipefail

SCRIPT_VERSION="1.2.3"
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
CACHE_AGENT="${QBIT_CACHE_AGENT:-/home/michael/dev/tools/silo/bin/qbit-cache-agent.py}"
CACHE_CLIENT_ID="$(basename "$0"):$$"
declare -a ALLOW_HASHES=()

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

# Print the dashboard panel to stdout.
print_dashboard() {
  local ts="$1"
  local checking="$2"
  local missing="$3"
  local moving="$4"
  local down="$5"
  local seeding_up="$6"
  local stopped_dl="$7"
  local stalled_up="$8"
  local uploading="$9"
  local stopped_up="${10}"
  local queued_up="${11}"
  local unexpected_down="${12}"
  local interval_s="${13}"

  local sep="──────────────────────"

  printf '── qBittorrent v%s ──\n' "$SCRIPT_VERSION"
  printf '%s\n' "$ts"
  printf '%s\n' "$sep"
  printf '%-12s : %5s\n' "checking" "$checking"
  printf '%-12s : %5s\n' "missing" "$missing"
  printf '%-12s : %5s\n' "moving" "$moving"
  printf '%-12s : %5s\n' "downloading" "$down"
  printf '%-12s : %5s\n' "seeding(up)" "$seeding_up"
  printf '%-12s : %5s\n' "stoppedDL" "$stopped_dl"
  printf '%-12s : %5s\n' "stalledUP" "$stalled_up"
  printf '%-12s : %5s\n' "uploading" "$uploading"
  printf '%-12s : %5s\n' "stoppedUP" "$stopped_up"
  printf '%-12s : %5s\n' "queuedUP" "$queued_up"
  printf '%s\n' "$sep"
  printf '%-12s : %5s\n' "unexpected↓" "$unexpected_down"
  printf '%s\n' "$sep"
  printf '%-12s : %4ss\n' "interval" "$interval_s"
}

_FORCE_REDRAW=0
_on_winch() { _FORCE_REDRAW=1; }
[[ "$DASHBOARD" -eq 1 ]] && trap '_on_winch' SIGWINCH

iteration=0

while true; do
  iteration=$((iteration + 1))

  FETCH_ERROR=""
  TORRENTS_JSON="[]"
  if [[ "$USE_CACHE" -eq 1 ]]; then
    _raw=""
    if ! _raw="$(QBIT_URL="$QBIT_URL" QBIT_USER="$QBIT_USER" QBIT_PASS="$QBIT_PASS" \
      python3 "$CACHE_AGENT" \
        --max-age "$CACHE_MAX_AGE" \
        --requested-interval "$INTERVAL_S" \
        --client-id "$CACHE_CLIENT_ID" \
        --ensure-daemon \
        2>/dev/null)"; then
      FETCH_ERROR="cache_fetch_failed"
    elif ! jq -e . >/dev/null 2>&1 <<<"$_raw"; then
      FETCH_ERROR="cache_invalid_json"
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
      printf '%s\n' "$_err_ts"
      printf '──────────────────────\n'
      printf '%-12s : %s\n' "error" "$FETCH_ERROR"
      printf '──────────────────────\n'
      printf '%-12s : %4ss\n' "interval" "$INTERVAL_S"
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

  read -r CHECKING MISSING MOVING DOWN UP COUNT_ZERO COUNT_PARTIAL STOPPED_UP STOPPED_DL STALLED_UP UPLOADING QUEUED_UP TOP_STATES <<<"$(jq -r '
    [
      ([.[] | (.state // "" | ascii_downcase) | select(startswith("checking"))] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "missingfiles")] | length),
      ([.[] | select((.state // "" | ascii_downcase) == "moving")] | length),
      ([.[] | select(
        (.state // "" | ascii_downcase) == "downloading"
        or (.state // "" | ascii_downcase) == "stalleddl"
        or (.state // "" | ascii_downcase) == "queueddl"
        or (.state // "" | ascii_downcase) == "forceddl"
        or (.state // "" | ascii_downcase) == "metadl"
      )] | length),
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
      "$CHECKING" "$MISSING" "$MOVING" "$DOWN" "$UP" \
      "$STOPPED_DL" "$STALLED_UP" "$UPLOADING" "$STOPPED_UP" "$QUEUED_UP" \
      "${#UNEXPECTED_DOWN[@]}" "$INTERVAL_S"
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

  sleep "$INTERVAL_S" &
  _SLEEP_PID=$!
  wait "$_SLEEP_PID" 2>/dev/null || true
  kill "$_SLEEP_PID" 2>/dev/null || true
done
