# silo Project Tracker

**Last updated:** 2026-03-26
**Reviewed against:** actual code in `bin/`, `config/`, `silo` dispatcher

---

## Reality Check: Prior TODOs vs. Actual Code

| # | Item | Prior Status | Actual Code State | Verdict |
|---|------|-------------|-------------------|---------|
| 1 | Bypass logic in `silo-cache-agent.py` / `silo-cache-daemon.py` | ✅ complete | Both call `exec_hashall_script(..., use_bypass=True)`; `silo_hashall_shared.py` clears password env vars when whitelist detected | ✅ DONE |
| 2 | Verify connection recovery after qBittorrent restart | ⚠ pending | Dashboard detects ban/reset (errno 104), but has **no automatic reconnect loop** — it prints an error and halts. No retry/re-login path exists. | ❌ NOT DONE — gap is real |
| 3 | SABnzbd error handling (port qBit ban detection) | 📋 proposal | `sab_api_request()` swallows all exceptions with a bare `except Exception: return {}`. No connection-reset detection, no ban warning, no retry. | ❌ NOT DONE |
| 4 | rTorrent integration | 📋 proposal | No rTorrent code anywhere in repo. Not started. | ❌ NOT STARTED |
| 5 | Consolidated `config/silo.yml` | 📋 proposal | Both dashboards already read `config/silo.yml` with `downloaders.qbittorrent` / `downloaders.sabnzbd` keys. Config file itself doesn't exist in repo (no committed example). | ✅ DONE (code), ⚠ missing example config |
| 6 | Unified `silo` CLI dispatcher | 📋 proposal | Root `silo` script dispatches `qbit` / `sab` / `sabnzbd` via `os.execve`. Works. | ✅ DONE |
| 7 | silo-dashboard.py input lag / non-blocking input | ⚠ open (archive doc) | `get_key()` uses blocking `select` with 0.1s timeout. No key-drain loop. Tab view still triggers full redraw. | ❌ NOT DONE |
| 8 | Async mediainfo (no blocking render-path calls) | ⚠ open (archive doc) | `get_mediainfo_summary_cached()` exists with background updater, but cache misses still block the render path synchronously. | ⚠ PARTIAL |
| 9 | Terminal resize (SIGWINCH) handling | ⚠ open (archive doc) | No `signal.signal(SIGWINCH, ...)` handler found. No resize detection. | ❌ NOT DONE |
| 10 | Example / committed `config/silo.yml` | implicit gap | Only `config/silo-filter-presets.yml` is present. No `config/silo.yml` example committed to repo. | ❌ MISSING |
| 11 | `silo` dispatcher: no `rtorrent` subcommand wired | design gap | `DASHBOARDS` dict has `qbit` and `sab`/`sabnzbd` only. `rtorrent` not present. | ❌ NOT WIRED (expected) |
| 12 | Duplicate `check_auth_bypass()` implementation | code smell | Identical function lives in both `silo_hashall_shared.py` and `silo-dashboard.py`. Not shared. | ⚠ TECH DEBT |

---

## Ranked TODO List

Priority ranking: **P1** = blocks correct behavior / data loss risk · **P2** = user-visible bug or reliability gap · **P3** = quality / maintainability · **P4** = future capability

### P1 — Correctness / Reliability

**P1-A: Connection recovery after qBittorrent restart**
- *Gap:* On ban or connection reset the dashboard prints an error and hangs/exits. No reconnect.
- *Fix:* Add a retry loop in the main poll cycle. After N consecutive failures, wait and re-attempt login using `check_auth_bypass` → `qbit_login`. Surface status in banner.
- *Files:* `bin/silo-dashboard.py` — main loop and `qbit_login` call site.

**P1-B: SABnzbd connection error surfacing**
- *Gap:* `sab_api_request()` eats all exceptions silently. User sees a blank/stale UI with no explanation when SABnzbd is unreachable.
- *Fix:* Distinguish `urllib.error.URLError` (network down), `json.JSONDecodeError` (bad response), and connection reset (errno 104). Surface in banner. Add simple retry counter.
- *Files:* `bin/silo-sabnzbd.py` — `sab_api_request()`.

### P2 — User-Visible UX / Responsiveness

**P2-A: Terminal resize (SIGWINCH) handling**
- *Gap:* Resizing the terminal leaves the TUI corrupted until a manual refresh key is pressed.
- *Fix:* Install `signal.signal(signal.SIGWINCH, handler)` to set a `needs_resize` flag; trigger full redraw on next loop iteration.
- *Files:* `bin/silo-dashboard.py`, `bin/silo-sabnzbd.py`.

**P2-B: silo-dashboard non-blocking input / key drain**
- *Gap:* Input lag when multiple keys are queued; each loop iteration only drains one key.
- *Fix:* Replace single `get_key()` call with a drain loop: read keys until `select` returns nothing, then process as batch.
- *Files:* `bin/silo-dashboard.py` — `get_key()` and main input loop.

**P2-C: MediaInfo cache misses block render**
- *Gap:* On cache miss, `mediainfo` is invoked synchronously in the render path, freezing the UI.
- *Fix:* Return `"…"` placeholder on miss; background thread enqueues the missing hash for immediate fetch.
- *Files:* `bin/silo-dashboard.py` — `get_mediainfo_summary_cached()` and its render call sites.

### P3 — Code Quality / Tech Debt

**P3-A: Deduplicate `check_auth_bypass()`**
- *Gap:* Identical function in `silo_hashall_shared.py` and `silo-dashboard.py`. Any fix must be applied twice.
- *Fix:* Import from `silo_hashall_shared` in the dashboard; remove local copy.
- *Files:* `bin/silo-dashboard.py`, `bin/silo_hashall_shared.py`.

**P3-B: Commit an example `config/silo.yml`**
- *Gap:* No committed example config. First-run UX is broken — user must guess the schema.
- *Fix:* Add `config/silo.yml.example` with commented-out stubs for both `downloaders.qbittorrent` and `downloaders.sabnzbd`.
- *Files:* `config/silo.yml.example` (new file).

**P3-C: Reconcile `docs/TODO.md` with reality**
- *Gap:* Several items marked as proposals are already done (unified dispatcher, consolidated config). Document is misleading.
- *Fix:* Archive old `TODO.md` content; point to this `TRACKER.md` as the canonical source.
- *Files:* `docs/TODO.md`.

**P3-D: Remove / document legacy `qbit-*` symlinks**
- *Gap:* `docs/migration/REFACTOR-TO-SILO.md` says symlinks are deprecated but no removal date is set.
- *Fix:* Set a deprecation deadline; add a warning print to the symlink targets; schedule removal.
- *Files:* `bin/qbit-*.py` symlinks.

### P4 — Future Capability

**P4-A: Unified multi-client dashboard (brainstorm → design → implement)**
- *Goal:* Single process, shared rendering primitives, tab/hotkey to switch clients without re-launching.
- *Status:* **Brainstorm task created** (Task #1 in this session). See design notes below.
- *Files:* TBD — likely `bin/silo-tui.py` (new) or refactor of `silo-dashboard.py`.

**P4-B: rTorrent integration**
- *Goal:* Skeleton dashboard for rTorrent; register `rtorrent` subcommand in `silo` dispatcher.
- *Dependency:* Blocks on P4-A design — rTorrent view should be built on the shared primitives.
- *Files:* `bin/silo-rtorrent.py` (new), `silo` dispatcher.

**P4-C: `silo` dispatcher: `--list` and `--help` polish**
- *Goal:* `silo --help` should show all subcommands with one-line descriptions; `silo --list` for scripting.
- *Files:* `silo`.

---

## Remediation Plan

### Phase 1 — Stability (P1 items, ship first)

1. **`silo-dashboard.py` reconnect loop** (P1-A)
   - Wrap the main poll tick in a connection-state machine: `CONNECTED` → `ERROR` → `RETRYING` → `CONNECTED`.
   - On `ConnectionResetError` or login failure: set `ERROR`, display banner `"Connection lost — retrying in Xs"`, count down, re-run `check_auth_bypass` → `qbit_login`.
   - Avoid hammering: use exponential backoff (2s, 5s, 15s cap).

2. **`silo-sabnzbd.py` error surfacing** (P1-B)
   - Expand `sab_api_request()` to categorize failures and return a structured result (or raise typed exceptions).
   - Caller surfaces message in banner using existing `set_banner()` equivalent.
   - Add connection-reset detection matching qBit pattern.

### Phase 2 — UX Polish (P2 items)

3. **SIGWINCH handler** (P2-A) — small, surgical, both files.
4. **Key drain loop** (P2-B) — replace `get_key()` call site with while-drain in `silo-dashboard.py`.
5. **MediaInfo placeholder** (P2-C) — guard the sync `mediainfo` call with a miss-returns-placeholder path.

### Phase 3 — Cleanup (P3 items)

6. **Deduplicate `check_auth_bypass`** (P3-A) — one-line import change.
7. **Commit `config/silo.yml.example`** (P3-B) — new file, no code change.
8. **Retire `docs/TODO.md`** (P3-C) — replace body with pointer to this file.
9. **Deprecation deadline for `qbit-*` symlinks** (P3-D).

### Phase 4 — Architecture (P4 items, design-first)

10. **Brainstorm unified multi-client design** (P4-A) — see Task #1. Output: design doc in `docs/design/unified-dashboard.md`.
11. **rTorrent skeleton** (P4-B) — implement after unified design is settled.
12. **Dispatcher polish** (P4-C) — minor, do alongside any dispatcher touch.

---

## Brainstorm Seed: Unified Multi-Client Dashboard (P4-A)

*This is a seed for Task #1. Expand in `docs/design/unified-dashboard.md`.*

**Problem:** Three separate TUI scripts (`silo-dashboard.py`, `silo-sabnzbd.py`, future `silo-rtorrent.py`) re-implement:
- Terminal setup / teardown (`tty.setraw`, `termios`, `atexit`)
- Color/ANSI rendering
- Key input loop (`get_key`, `select`)
- Banner / status line
- Paging, filtering, tab display
- Config loading from `config/silo.yml`
- Connection auth (partially shared via `silo_hashall_shared.py`)
- Error handling and reconnect

**Design questions to answer:**
1. **Single process vs. exec-replace:** Current dispatcher `os.execve`s into each script (process replace). Unified view requires a single long-running process.
2. **Shared rendering layer:** Extract `draw_*`, `ColorScheme`, banner, paging into `bin/silo_tui_core.py`. Each client provides a `render_rows()` callback.
3. **Client abstraction:** Define a `DownloaderClient` protocol: `connect()`, `fetch()`, `action(cmd, id)`, `disconnect()`. Each client (qBit, SAB, rTorrent) implements it.
4. **Client switching UX:** Hotkey (e.g., `Ctrl-W` or `F1`/`F2`/`F3`) cycles active client panel without leaving the TUI. Title bar shows active client name.
5. **Config:** `config/silo.yml` already has the right shape (`downloaders.*`). Extend to include display name and enabled flag per client.
6. **Cache protocol:** qBit uses a hashall-backed cache. SABnzbd polls directly. Unified layer could abstract this behind a `fetch()` method.
7. **Incremental migration path:** Keep existing scripts working; new `silo-tui.py` imports shared core and client modules. Register as `silo tui` subcommand initially.
