# silo Project Tracker

**Last updated:** 2026-03-28
**Reviewed against:** actual code in `bin/`, `config/`, `silo` dispatcher

---

## Completed This Session (2026-03-26 → 2026-03-28)

| Item | Status |
|------|--------|
| rTorrent integration (`silo_client_rt.py`, in-dashboard `\` switcher) | ✅ DONE |
| rTorrent enriched state inference (stalledUP, checkingDL/UP, metaDL, stoppedUP) | ✅ DONE |
| rTorrent cache daemon (`silo-rt-cache-daemon.py`, `silo_cache_common.py`) | ✅ DONE |
| Cache header panel switches with active client | ✅ DONE |
| qBittorrent connection recovery (backoff + re-login) | ✅ DONE |
| SABnzbd typed error handling in `sab_api_request()` | ✅ DONE |
| `qbit-*` shims: symlinks → standalone files with deprecation warnings | ✅ DONE |
| `silo` dispatcher: `--help`, `--list`, unknown-subcommand error | ✅ DONE |
| SABnzbd client module (`silo_client_sab.py`) | ✅ DONE |
| `silo-sabnzbd.py` refactored to import from `silo_client_sab` | ✅ DONE |
| SABnzbd wired as third client in `silo-dashboard.py` (\ cycles qBit→rt→SAB) | ✅ DONE |
| `config/silo.yml.example` updated with SABnzbd fields and cycle hotkey | ✅ DONE |
| `config/silo.yml` updated with SABnzbd api_url | ✅ DONE |
| Example config `config/silo.yml.example` committed | ✅ DONE (prior session) |

---

## Reality Check: Prior TODOs vs. Actual Code

| # | Item | Status |
|---|------|--------|
| 1 | Bypass logic in `silo-cache-agent.py` / `silo-cache-daemon.py` | ✅ DONE |
| 2 | Connection recovery after qBittorrent restart | ✅ DONE — backoff + re-login in dashboard |
| 3 | SABnzbd error handling (typed errors, connection-reset detection) | ✅ DONE — `silo_client_sab.request()` |
| 4 | rTorrent integration | ✅ DONE — `silo_client_rt.py` + in-dashboard switch |
| 5 | Consolidated `config/silo.yml` | ✅ DONE |
| 6 | Unified `silo` CLI dispatcher | ✅ DONE |
| 7 | silo-dashboard.py non-blocking input / key drain | ✅ DONE (prior session) |
| 8 | Async mediainfo (no blocking render-path calls) | ⚠ PARTIAL |
| 9 | Terminal resize (SIGWINCH) handling | ✅ DONE (prior session) |
| 10 | Example `config/silo.yml.example` committed | ✅ DONE |
| 11 | `rtorrent` subcommand in `silo` dispatcher | ✅ DONE |
| 12 | Duplicate `check_auth_bypass()` | ⚠ TECH DEBT (won't-fix: different semantics) |

---

## Ranked TODO List

Priority: **P1** = correctness / data loss · **P2** = user-visible UX · **P3** = quality · **P4** = future

### P1 — Correctness / Reliability

*No open P1 items.*

### P2 — User-Visible UX

**P2-A: MediaInfo cache misses block render**
- *Gap:* On cache miss, `mediainfo` is invoked synchronously in the render path, freezing the UI.
- *Fix:* Return `"…"` placeholder on miss; background thread enqueues the missing hash for immediate fetch.
- *Files:* `bin/silo-dashboard.py` — `get_mediainfo_summary_cached()` and its render call sites.

**P2-B: SABnzbd in dashboard — scope filtering (w/u/v/e/g) has no effect**
- *Gap:* Scope keys filter by `raw.state` which is a qBit field; SABnzbd raw dicts use native SABnzbd field names.
- *Impact:* Pressing `w`/`u`/`v`/`e`/`g` while on SABnzbd shows nothing (all rows filtered out).
- *Fix:* Map SABnzbd `state` (from unified schema top level) into the scope filter check, or disable scope keys when `active_client == "sabnzbd"`.
- *Files:* `bin/silo-dashboard.py` — `rows_to_render` filter block.

**P2-C: rTorrent in dashboard — scope filtering same issue as P2-B**
- *Fix:* Same approach — use row-level `state` for non-qBit clients.

### P3 — Code Quality / Tech Debt

**P3-A: Deduplicate `check_auth_bypass()`**
- *Gap:* Identical function in `silo_hashall_shared.py` and `silo-dashboard.py`. Won't-fix for now (different semantics: dashboard version is opener/cookie-jar aware).
- *Status:* Deferred.

**P3-B: `qbit-*` shim removal (2026-07-01)**
- All `qbit-*` shims print deprecation warnings. Scheduled removal 2026-07-01.
- *Files:* `bin/qbit-cache-agent.py`, `bin/qbit-cache-daemon.py`, `bin/qbit-dashboard.py`.

**P3-C: `silo_client_sab.py` — `last_error` cleared when queue fails but history succeeds**
- *Gap:* If queue fetch fails but history succeeds within `fetch()`, `last_error` is cleared by the history success. Queue error is silently lost.
- *Impact:* Very unlikely in practice (same endpoint, same credentials). Low priority.
- *Fix:* Accumulate errors across both calls rather than overwriting.

### P4 — Future Capability

**P4-A: SABnzbd in dashboard — actions beyond P/D/T**
- Change category (`C`), mark-as-completed (`M`) not wired in the dashboard `ACTIONS` table.
- These are available in `silo-sabnzbd.py` via `apply_action()` but not surfaced in the unified dashboard.
- *Files:* `bin/silo_client_sab.py` ACTIONS, `bin/silo-dashboard.py` action dispatch.

**P4-B: rTorrent — SCGI socket transport**
- Current rTorrent connection requires an HTTP proxy in front of the SCGI socket.
- Direct SCGI support would remove the nginx dependency.
- *Files:* `bin/silo_client_rt.py` — add `SCGITransport` class.

**P4-C: SABnzbd cache daemon**
- SABnzbd currently fetches direct-API every 5s from the dashboard.
- A `silo-sab-cache-daemon.py` on the `silo_cache_common` pattern would allow multiple dashboard instances and reduce API load.
- *Dependency:* Only worth it if multiple dashboard instances run simultaneously.

---

## Architecture: Multi-Client Dashboard

The unified client architecture is now live:

```
silo-dashboard.py
  ├── silo_client_rt.py    (rTorrent: xmlrpc.client, connect/fetch/ACTIONS)
  └── silo_client_sab.py   (SABnzbd: urllib, connect/fetch/ACTIONS)

silo-sabnzbd.py            (standalone TUI — imports silo_client_sab)

silo-rt-cache-daemon.py    (rTorrent background poller)
silo_cache_common.py       (generic daemon infrastructure)
```

**Client module protocol** (all clients expose the same surface):
- `NAME: str`, `KEY: str`
- `last_error: str` (module-level, cleared on success)
- `connect(url, key) -> Conn`
- `fetch(conn, ...) -> list[dict]` — unified silo schema
- `ACTIONS: dict[str, tuple[str, callable]]` — key → (label, fn(conn, id, item) → str)

**Unified schema keys** (same for qBit, rTorrent, SABnzbd):
`name, save_path, nohl, state, st, progress, progress_pct, size, ratio,
dlspeed, upspeed, seeds, peers, eta, added, added_short, tracker,
category, tags, hash, source, raw`

`raw` always contains `dlspeed` and `upspeed` as `int` (bytes/s) for header bandwidth totals.

**Client cycling** (`\` key):
```
qBit → rTorrent (if configured) → SABnzbd (if configured) → qBit
```
Unconfigured clients are skipped silently.
