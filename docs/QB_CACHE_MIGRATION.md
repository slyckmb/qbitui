# qBittorrent Cache Migration: hashall → silo

**Status:** Planned / In Progress  
**Session:** `silo-20260422-054758-claude`  
**Branch:** `cr/silo-20260422-054758-claude`

---

## Problem

The qBittorrent cache daemon and agent live in the hashall repo
(`src/hashall/qb_cache.py`, `bin/qb-cache-agent.py`, `bin/qb-cache-daemon.py`).
Silo consumes them via 9-line exec-shims that locate hashall at runtime via
`silo_hashall_shared.exec_hashall_script()`.

This is backwards:
- The RT cache daemon already lives natively in silo
- qB daemon fixes travel through hashall's branch/merge cycle before silo picks them up
- Cross-repo path resolution fails in worktree deployments
- `requests` library dependency pulled in transitively through hashall

## Outcome

Silo owns both caches. hashall becomes a read-only consumer of the JSON files the
silo-managed daemon writes. The cross-repo shim layer is eliminated.

---

## Files Changed

### Created
| File | Description |
|---|---|
| `bin/qb_cache_lib.py` | Native silo copy of `hashall/src/hashall/qb_cache.py` (adapted) |

### Replaced (shims → real implementations)
| File | From | To |
|---|---|---|
| `bin/silo-cache-agent.py` | 9-line exec shim | Entry point importing `agent_main` from `qb_cache_lib` |
| `bin/silo-cache-daemon.py` | 9-line exec shim | Entry point importing `daemon_main` from `qb_cache_lib` |

### Modified
| File | Change |
|---|---|
| `bin/silo-dashboard.py` line 31 | `from silo_hashall_shared import DEFAULT_HASHALL_CACHE_BASE` → `from qb_cache_lib import DEFAULT_QB_CACHE_BASE` |
| `bin/silo_cache_common.py` | Generalize discovery functions with `daemon_name` + `env_var` params |

### Deleted
| File | Reason |
|---|---|
| `bin/silo_hashall_shared.py` | Dead code after migration |
| `bin/qbit-cache-agent.py` | Already deprecated (removal date 2026-07-01), accelerated |
| `bin/qbit-cache-daemon.py` | Already deprecated (removal date 2026-07-01), accelerated |

---

## Step 0: Git History Import

**Why:** Preserve the 7-commit hashall lineage for each file so `git log bin/qb_cache_lib.py`
shows full context. History is imported before any adaptation edits so subsequent commits
appear on top of the hashall ancestry.

### Rename history (from `git log --follow --diff-filter=R`)

| File | Rename |
|---|---|
| `src/hashall/qb_cache.py` | None — created directly |
| `bin/qb-cache-agent.py` | Renamed from `bin/qbit-cache-agent.py` at commit `940df09` |
| `bin/qb-cache-daemon.py` | Renamed from `bin/qbit-cache-daemon.py` at commit `940df09` |

### Import procedure

```bash
# Clone hashall to temp location
git clone /home/michael/dev/work/hashall /tmp/hashall-export
cd /tmp/hashall-export

# Strip to 3 files with silo path remapping
git filter-repo \
  --path src/hashall/qb_cache.py \
  --path bin/qb-cache-agent.py \
  --path bin/qb-cache-daemon.py \
  --path-rename 'src/hashall/qb_cache.py:bin/qb_cache_lib.py' \
  --path-rename 'bin/qb-cache-agent.py:bin/silo-cache-agent.py' \
  --path-rename 'bin/qb-cache-daemon.py:bin/silo-cache-daemon.py'

# Merge filtered history into silo worktree
cd /home/michael/dev/tools/silo/.agent/worktrees/silo-20260422-054758-claude
git remote add hashall-export /tmp/hashall-export
git fetch hashall-export
git merge hashall-export/main --allow-unrelated-histories \
  -m "chore: import qb cache history from hashall (path-remapped)"
git remote remove hashall-export

# Verify
git log --oneline bin/qb_cache_lib.py | head -10
```

---

## Step 1: Adapt `bin/qb_cache_lib.py`

Starting from the hashall-imported version, apply these targeted changes:

### a. Remove hashall import, replace `_fetch_torrents_snapshot`

Delete: `from hashall.qbittorrent import get_qbittorrent_client`

Replace `_fetch_torrents_snapshot` with stdlib-only implementation using
`urllib.request` / `http.cookiejar` (same pattern as silo-dashboard.py's
`make_opener`, `qbit_login`, `qbit_request`):

```python
def _fetch_torrents_snapshot(*, qbit_url, username, password):
    import urllib.request, urllib.parse, http.cookiejar
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    data = urllib.parse.urlencode({"username": username, "password": password}).encode()
    opener.open(f"{qbit_url}/api/v2/auth/login", data, timeout=10)
    def _get(path):
        return opener.open(f"{qbit_url}/api/v2/{path}", timeout=10).read().decode()
    qb_version = _get("app/version").strip()
    api_version = _get("app/webapiVersion").strip()
    qb_profile = {"client_version": qb_version, "api_version": api_version}
    raw = _get("torrents/info?sort=name")
    torrents = json.loads(raw)
    return json.dumps(torrents, indent=2), len(torrents), qb_profile, {"mode": "not_enriched"}
```

### b. Add module-level cache constant

```python
DEFAULT_QB_CACHE_BASE = Path.home() / ".cache" / "silo-qb"
LEGACY_QB_CACHE_BASE = Path.home() / ".cache" / "hashall-qb"
```

Replace all inline `Path.home() / ".cache" / "hashall-qb"` in both parsers with
the silo-owned default. Keep the legacy path only as a read fallback while
downstream repos migrate.

### c. Fix `--daemon-cmd` default in `build_agent_parser`

Old: `Path(__file__).resolve().parents[2] / "bin" / "qb-cache-daemon.py"` (hashall layout)  
New: `_QB_DAEMON_DEFAULT_STR` (set at module level via `discover_daemon_script()`)

### d. Version and docstring

- `__version__` / `SEMVER` → `"1.0.0"`
- Update module docstring to remove hashall references
- Confirm zombie fix is present: `if args.max_age <= 0: return 0` in `agent_main()`

---

## Step 2: Replace `bin/silo-cache-agent.py`

```python
#!/usr/bin/env python3
"""silo-cache-agent — qBittorrent shared cache agent (silo-native implementation)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from qb_cache_lib import agent_main
sys.exit(agent_main())
```

---

## Step 3: Replace `bin/silo-cache-daemon.py`

```python
#!/usr/bin/env python3
"""silo-cache-daemon — qBittorrent shared cache daemon (silo-native implementation)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from qb_cache_lib import daemon_main
sys.exit(daemon_main())
```

Note: the docstring must contain `silo-cache-daemon` for the discovery validator.

---

## Step 4: Update `bin/silo-dashboard.py`

Line 31 change:
```python
# Before:
from silo_hashall_shared import DEFAULT_HASHALL_CACHE_BASE
# After:
from qb_cache_lib import DEFAULT_QB_CACHE_BASE
```

The dashboard uses the silo-owned cache base directly.

---

## Step 5: Delete dead files

```
bin/silo_hashall_shared.py
bin/qbit-cache-agent.py
bin/qbit-cache-daemon.py
```

---

## Step 6: Port RT Cache Hardening

Generalize `silo_cache_common.py` discovery functions to support qB daemon:

### 6a. Add `daemon_name` + `env_var` parameters

```python
def _validate_daemon_script(path: Path, daemon_name: str = "silo-rt-cache-daemon") -> bool: ...
def _find_running_daemon(daemon_name: str = "silo-rt-cache-daemon") -> Path | None: ...
def discover_daemon_script(
    default_relative: Path,
    daemon_name: str = "silo-rt-cache-daemon",
    env_var: str = "SILO_RT_DAEMON_SCRIPT",
) -> Path | None: ...
```

Existing RT call sites: add `daemon_name="silo-rt-cache-daemon"` + `env_var="SILO_RT_DAEMON_SCRIPT"` (no behavior change).

### 6b. Wire discovery into `qb_cache_lib.py`

At module level, resolve the default daemon path:

```python
_bin_dir = Path(__file__).resolve().parent
try:
    import sys as _sys; _sys.path.insert(0, str(_bin_dir))
    from silo_cache_common import discover_daemon_script as _discover_daemon
    _discovered = _discover_daemon(
        default_relative=_bin_dir / "silo-cache-daemon.py",
        daemon_name="silo-cache-daemon",
        env_var="SILO_QB_DAEMON_SCRIPT",
    )
except ImportError:
    _discovered = None
_QB_DAEMON_DEFAULT_STR = str(_discovered or (_bin_dir / "silo-cache-daemon.py"))
```

Use `_QB_DAEMON_DEFAULT_STR` as `default=` for `--daemon-cmd` in `build_agent_parser()`.

### Environment variables

| Daemon | Env var override |
|---|---|
| RT: `silo-rt-cache-daemon.py` | `SILO_RT_DAEMON_SCRIPT` |
| qB: `silo-cache-daemon.py` | `SILO_QB_DAEMON_SCRIPT` |

---

## Verification

```bash
# History check
git log --oneline bin/qb_cache_lib.py | wc -l   # expect >= 7

# Syntax
python3 -m py_compile bin/qb_cache_lib.py
python3 -m py_compile bin/silo-cache-agent.py
python3 -m py_compile bin/silo-cache-daemon.py
python3 -m py_compile bin/silo-dashboard.py
python3 -m py_compile bin/silo_cache_common.py

# Daemon smoke test
python3 bin/silo-cache-daemon.py --once
ls -lah ~/.cache/silo-qb/torrents-info.json

# Agent smoke test
python3 bin/silo-cache-agent.py --max-age 30 | python3 -m json.tool | head -5

# Discovery smoke test
python3 -c "
import sys; sys.path.insert(0,'bin')
from silo_cache_common import discover_daemon_script
from pathlib import Path
for name, daemon, env in [
    ('RT', 'silo-rt-cache-daemon.py', 'SILO_RT_DAEMON_SCRIPT'),
    ('qB', 'silo-cache-daemon.py', 'SILO_QB_DAEMON_SCRIPT'),
]:
    p = discover_daemon_script(Path('bin') / daemon, daemon_name=daemon.replace('.py',''), env_var=env)
    print(f'{name}: {p}')
"

# Shim gone
python3 -c "import sys; sys.path.insert(0,'bin'); import silo_hashall_shared" \
  && echo FAIL || echo OK

# Tests
python3 -m pytest tests/ -v
```

---

## Critical Facts

1. **Cache dir is `~/.cache/silo-qb`** — `~/.cache/hashall-qb` is read fallback only.
2. **Zombie fix present** — `if args.max_age <= 0: return 0` in `agent_main()` (~line 328).
3. **Dashboard default** — `--cache-base-dir` defaults to `DEFAULT_QB_CACHE_BASE`.
4. **Don't touch hashall** — cleanup of hashall copies is a separate PR.
5. **History import before adaptation** — Step 0 must precede Step 1.
6. **`silo-cache-daemon.py` docstring must contain `silo-cache-daemon`** — used by validator.
7. **Env vars are independent** — `SILO_RT_DAEMON_SCRIPT` for RT, `SILO_QB_DAEMON_SCRIPT` for qB.
