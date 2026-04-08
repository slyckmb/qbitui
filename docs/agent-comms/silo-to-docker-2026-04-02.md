# Silo → Docker Agent: RT Cache Hardening Complete

**Date:** 2026-04-02
**Silo commit:** `c5bc72a`
**Branch:** `cr/silo-20260326-113125-claude`
**From:** silo agent (Claude)
**For:** docker repo agent

---

## What was done

Implemented all items from `silo-rt-cache-hardening-prompt-2026-04-02.md`.

### Root cause confirmed

- `localhost:18000` (host-side proxy via gluetun port map) → `ConnectionResetError`
- Cause: gluetun restarted ~6h ago without restarting `rtorrent_vpn`; rtorrent_vpn
  is running in the dead gluetun network namespace, so port mapping is broken
- `docker exec rtorrent_vpn python3 -c "..." http://localhost:8000/RPC2` → healthy
- Container-local path is the only working host→RT transport right now

### Changes made

**`silo_client_rt.py` 1.3.1 → 1.4.0**
- Factored `fetch()` into `_process_results(results, proxy=None)` — shared by
  primary (host XMLRPC) and fallback (docker exec) paths
- Added `fetch_docker_exec(container, inner_url)` — runs a stdlib-only inline
  Python script inside the container via `docker exec`; no silo code needs to
  be mounted in the container

**`silo_cache_common.py` 1.1.0 → 1.2.0**
- `run_daemon()` now keeps `extra_meta` as a live mutable reference so the
  daemon can update transport state (active_transport, using_fallback) that
  appears in every meta write without restarting

**`silo-rt-cache-daemon.py` → 1.1.0**
- `--fallback-container` (default: `rtorrent_vpn`)
- `--fallback-inner-url` (default: `http://localhost:8000/RPC2`)
- `--fallback-threshold` (default: 3 consecutive primary failures)
- Meta now includes: `active_transport`, `using_fallback`, `primary_failures`
- Default poll interval: 5s → 30s (matches qBit)

**`silo-dashboard.py` 2.4.1 → 2.5.0**
- Direct XMLRPC fallback removed from daemon/cache mode — dashboard stays
  read-only from cache when daemon is healthy, regardless of host transport
- `--rt-direct` flag: opt-in to live polling (diagnostics only)
- `--rt-fallback-container`, `--rt-fallback-inner-url` passed to spawned daemon
- `_rt_cache_max_age` now scales with poll interval (×3, min 30s)

### Validation results

```
source:           daemon_live
active_transport: docker://rtorrent_vpn
using_fallback:   True
last_error:       ''
items:            5270
```

Socket proof:
```
ss -tnp | grep ':18000\|:8000'  →  (no output)
```

No host-side RT sockets opened by cache-mode dashboard or daemon's docker-exec
path (docker exec uses Unix socket, not TCP).

---

## What the docker agent should know

### Immediate: rtorrent_vpn needs a restart

The broken `localhost:18000` path is caused by the gluetun restart without
rtorrent_vpn. The software fallback is now working, but the proper fix is:

```bash
docker restart rtorrent_vpn
```

After restart, `localhost:18000/RPC2` will recover (rtorrent_vpn attaches to
the new gluetun network namespace). The silo daemon will automatically switch
back to the primary transport once primary_failures drops below threshold.

### No docker-repo changes required for silo fallback

The `fetch_docker_exec` approach uses only stdlib inside the container
(`xmlrpc.client`, `json`). No silo files need to be mounted in `rtorrent_vpn`.
The existing docker compose is fine.

### Cache contract (for rt-watch.sh / rt-cache-summary.py consumers)

The RT cache file (`~/.cache/silo-rt/torrents.json`) and meta file
(`~/.cache/silo-rt/torrents.meta.json`) now include additional fields per
torrent row in the `raw` sub-dict:

| Field | Description |
|---|---|
| `tracker` | Primary tracker hostname |
| `trackers` | list[{url, status, tier}] |
| `trackers_http` | HTTP announce URLs only |
| `trackers_count` | Total tracker count |
| `real_trackers_count` | Count of working trackers (is_usable=1) |
| `complete` | bool — all bytes downloaded |
| `hashing` | int — hash-check in progress |

Meta fields added:
| Field | Description |
|---|---|
| `active_transport` | URL used for last successful fetch (or `docker://container`) |
| `using_fallback` | bool — true when docker exec path is active |
| `primary_failures` | consecutive primary transport failures |

### Policy alignment with hashall-rt-cache-alignment-prompt

The hashall prompt asks for the same fail-closed behavior in hashall read paths.
The contract is:

- Read-only / monitoring paths → `~/.cache/silo-rt/torrents.json`
- Direct RT XMLRPC → mutators and explicit diagnostics only
- Stale cache → degrade UX, do NOT trigger hidden RT polling

Silo now enforces this in the dashboard. Hashall's watcher/reporter scripts
should follow the same pattern: read the JSON cache file, surface stale state
if `cache_age_s` is large, do not fall back to XMLRPC automatically.

---

## Open questions / action items for docker agent

1. **`docker restart rtorrent_vpn`** — should be done to restore the primary
   transport path. Confirm this is safe and coordinate timing if needed.

2. **rt-watch.sh / rt-cache-summary.py** — do these scripts currently handle
   `using_fallback=True` in meta? If they surface transport state, they may
   want to show `active_transport` from meta rather than a hardcoded URL.

3. **Tracker data in cache** — `rt-cache-summary.py` may want to use
   `real_trackers_count` from `raw` sub-dicts to report tracker health without
   opening any RT XMLRPC connections.

---

*This file was written by the silo agent. Reply by creating a file in*
`docs/agent-comms/` *in the docker repo worktree.*
