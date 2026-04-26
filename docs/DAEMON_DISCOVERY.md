# rTorrent Daemon Discovery

## Overview

The silo dashboard (`silo-dashboard`) uses an rTorrent cache daemon to efficiently poll and cache torrent data, reducing load on the rTorrent server. The daemon discovery system makes this flexible and robust across different deployment scenarios, including:

- **Main repository deployment**: Dashboard and daemon in `/home/michael/dev/tools/silo/bin/`
- **Worktree scenarios**: Dashboard in main repo, daemon in temporary worktree
- **Custom deployments**: Daemon in non-standard locations
- **Graceful degradation**: Dashboard works even if daemon is not found (falls back to direct API calls)

## Discovery Order (Priority Chain)

When the dashboard starts, it searches for the daemon script in this order:

### 1. Environment Variable Override

If `SILO_RT_DAEMON_SCRIPT` is set, the dashboard uses that path:

```bash
export SILO_RT_DAEMON_SCRIPT=/path/to/silo-rt-cache-daemon.py
silo-dashboard
```

This is the highest priority and overrides all other discovery methods. The specified file must:
- Exist and be readable
- Contain the marker `silo-rt-cache-daemon` in its first 500 characters
- Start with a Python shebang (`#!/usr/bin/env python3` or equivalent)

### 2. Running Process Detection

If no environment variable is set, the dashboard scans the system's process list (`/proc`) for an already-running daemon. If found, it extracts the daemon script path from the running process.

**How it works:**
- Scans `/proc/*/cmdline` for active processes containing `silo-rt-cache-daemon.py`
- Extracts the script path from the running command
- Uses that path if validation passes

**Advantages:**
- Fully automatic; no configuration needed
- Works seamlessly when daemon is running elsewhere (e.g., in another worktree)
- Ideal for "just works" deployments

**Example scenario:**
```
Worktree 1 starts daemon at:
  /home/michael/dev/tools/silo/.agent/worktrees/silo-20260421-070929-gemini/bin/silo-rt-cache-daemon.py

Worktree 2 (or main repo) runs dashboard:
  $ silo-dashboard
  [auto-discovered daemon at /home/michael/dev/tools/silo/.agent/worktrees/silo-20260421-070929-gemini/bin/silo-rt-cache-daemon.py]
```

### 3. Default Relative Path

The dashboard then checks for the daemon relative to itself:

```
/path/to/silo-dashboard.py → /path/to/silo-rt-cache-daemon.py
```

This is the original behavior and handles the common case where dashboard and daemon are in the same `bin/` directory.

### 4. Common Alternate Paths

As a final fallback, the dashboard checks these locations:

- `~/dev/tools/silo/bin/silo-rt-cache-daemon.py` (home directory reference)
- `.agent/worktrees/*/bin/silo-rt-cache-daemon.py` (any worktree in current repo)

## Use Cases and Examples

### Standard Deployment (No Configuration)

Dashboard and daemon in the same directory:

```bash
cd /home/michael/dev/tools/silo/bin
./silo-dashboard
# Dashboard auto-discovers silo-rt-cache-daemon.py in the same directory
```

**Result:** ✓ Works automatically (default path)

---

### Worktree with Shared Daemon

Dashboard in main repo, daemon in a worktree:

```bash
# Terminal 1: Start daemon in worktree
cd /home/michael/dev/tools/silo/.agent/worktrees/silo-20260421-070929-gemini/bin
./silo-rt-cache-daemon.py --xmlrpc-url http://localhost:18000/RPC2

# Terminal 2: Run dashboard in main repo
cd /home/michael/dev/tools/silo/bin
./silo-dashboard
# Dashboard auto-discovers running daemon (process detection)
```

**Result:** ✓ Works automatically (process auto-detection)

---

### Custom Daemon Location

Daemon in a non-standard location:

```bash
export SILO_RT_DAEMON_SCRIPT=/opt/silo/daemon/silo-rt-cache-daemon.py
silo-dashboard
```

**Result:** ✓ Works with environment variable override

---

### Explicit Worktree Path

Daemon in a specific worktree (when process detection doesn't work):

```bash
export SILO_RT_DAEMON_SCRIPT=/home/michael/dev/tools/silo/.agent/worktrees/silo-20260421-070929-gemini/bin/silo-rt-cache-daemon.py
silo-dashboard
```

**Result:** ✓ Works with environment variable

---

### Daemon Not Running

If the daemon script cannot be found anywhere:

```bash
cd /home/michael/dev/tools/silo/bin
./silo-dashboard
# Daemon discovery fails (returns None)
# Dashboard falls back to direct XML-RPC calls
```

**Result:** ✓ Dashboard still works, but queries rTorrent directly (no caching)

---

## Validation

When a daemon script path is discovered (from any source), the dashboard validates it:

✓ File exists and is readable  
✓ File has content (size > 0)  
✓ File starts with a Python shebang  
✓ File contains `silo-rt-cache-daemon` marker in docstring/header  

If validation fails, the dashboard tries the next discovery method.

## Debugging

### Check Which Daemon Was Discovered

The discovery process is silent by default. To see which daemon path was selected, check the daemon log:

```bash
cat ~/.cache/silo-rt/daemon.log | tail -20
```

Look for lines like:
```
[2026-04-22T12:34:56.789Z] starting daemon: /usr/bin/python3 /path/to/silo-rt-cache-daemon.py ...
```

### Verify Daemon is Running

```bash
# Check if daemon is running
ps aux | grep silo-rt-cache-daemon

# Check daemon status
cat ~/.cache/silo-rt/daemon.pid
cat ~/.cache/silo-rt/torrents.meta.json | jq '.daemon_running'
```

### Force Environment Variable Discovery

Explicitly set the daemon path to debug discovery:

```bash
export SILO_RT_DAEMON_SCRIPT=/path/to/daemon
silo-dashboard
```

If the dashboard still can't find the daemon, check:
- File path is absolute or properly expanded
- File exists: `ls -la /path/to/daemon`
- File is readable: `cat /path/to/daemon | head -1` (should show shebang)
- File contains marker: `grep -q "silo-rt-cache-daemon" /path/to/daemon && echo "Found"`

## Fallback Behavior

If the daemon script is not found **at any stage**, the dashboard gracefully degrades:

- **Cache layer is disabled** — dashboard skips daemon-based caching
- **Direct XML-RPC polling** — queries rTorrent directly on every redraw
- **Performance impact** — slower responses, more load on rTorrent
- **No errors** — dashboard still renders normally (just without caching)

This allows the dashboard to continue functioning in minimal-configuration environments.

## Technical Details

### Discovery Function

The discovery is implemented in `silo_cache_common.py`:

```python
discover_daemon_script(default_relative: Path) -> Path | None:
    """
    Discover silo-rt-cache-daemon.py by trying:
      1. SILO_RT_DAEMON_SCRIPT environment variable
      2. Running process detection (/proc scan)
      3. Default relative path
      4. Common alternate paths
    
    Returns Path if found, None if all strategies fail (graceful).
    """
```

### Validation Helper

Validation is performed by `_validate_daemon_script(script_path: Path) -> bool`:

```python
def _validate_daemon_script(script_path: Path) -> bool:
    """Check: exists, readable, Python shebang, contains marker."""
```

### Process Detection Helper

Process scanning is implemented in `_find_running_daemon() -> Path | None`:

```python
def _find_running_daemon() -> Path | None:
    """Scan /proc for running daemon, extract script path."""
```

## Troubleshooting

### Problem: "Daemon script not found"

**Cause:** None of the discovery methods found a valid daemon script.

**Solution:**
1. Check if daemon is actually running: `ps aux | grep silo-rt-cache-daemon`
2. If running, set `SILO_RT_DAEMON_SCRIPT` explicitly to its path
3. If not running, start it: `/path/to/silo-rt-cache-daemon.py &`

### Problem: Wrong daemon discovered

**Cause:** Discovery found a different daemon than intended (e.g., from a different worktree).

**Solution:**
Set `SILO_RT_DAEMON_SCRIPT` to override:
```bash
export SILO_RT_DAEMON_SCRIPT=/intended/path/silo-rt-cache-daemon.py
silo-dashboard
```

### Problem: Daemon crashes immediately after starting

**Cause:** Script validation passed, but daemon failed to run.

**Solution:**
1. Check daemon log: `cat ~/.cache/silo-rt/daemon.log`
2. Try running daemon manually to see errors: `/path/to/silo-rt-cache-daemon.py --xmlrpc-url http://localhost:18000/RPC2`
3. Check Python dependencies: `python3 -c "import silo_cache_common"`

## Summary

| Scenario | Discovery Method | Configuration |
|----------|------------------|---|
| Standard (same bin/) | Default relative path | None |
| Worktree with running daemon | Process detection | None |
| Custom location | Environment variable | `SILO_RT_DAEMON_SCRIPT=/path` |
| No daemon found | (None) → graceful fallback | N/A |

The system is designed to be **zero-configuration** for standard deployments while remaining flexible for advanced use cases.
