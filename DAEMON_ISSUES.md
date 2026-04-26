# rTorrent Daemon Issues & Root Cause Analysis

**Date:** 2026-04-22  
**Status:** CRITICAL — Daemon non-functional  
**Impact:** Cache layer disabled; dashboard falls back to direct API calls  

## Issue Summary

The rTorrent cache daemon (`silo-rt-cache-daemon.py`) is running but failing to fetch torrents, resulting in an empty cache. The daemon reports:

```
last_error: "rTorrent returned empty result from http://localhost:18000/RPC2"
primary_failures: 13
consecutive_failures: 13
```

## Root Cause

**`d.timestamp.load` field does not exist in rTorrent 0.16.5**

The silo_client_rt module requests this field:
```python
_MULTICALL_FIELDS = [
    "d.hash=",
    ...
    "d.timestamp.load=",  # ← Unsupported in rTorrent 0.16.5
    ...
]
```

When `d.multicall.filtered()` is called with this field, rTorrent 0.16.5 raises:
```
Fault -500: 'Command "d.timestamp.load" does not exist.'
```

This exception is caught by `silo_client_rt.fetch()` → returns empty list → daemon treats as "empty result" error.

## Technical Timeline

- **2026-03-28**: Commit `f7a5ba4` added `d.timestamp.load` field for "Added" column
  - Works on newer rTorrent versions (0.17+)
  - **Breaks on rTorrent 0.16.5** (incompatible)
- **2026-04-09 onwards**: Daemon repeatedly fails with empty results
- **2026-04-21 onwards**: Dashboard cannot fetch torrents via cache layer

## Environment

- **rTorrent version**: 0.16.5
- **XMLRPC endpoint**: http://localhost:18000/RPC2 ✓ Responding
- **Daemon location**: /home/michael/dev/tools/silo/bin/silo-rt-cache-daemon.py
- **Cache directory**: ~/.cache/silo-rt/
- **Cache data file**: EMPTY (never successfully written)

## Verification Steps

### 1. Confirm rTorrent Version
```bash
$ curl -s -X POST http://localhost:18000/RPC2 \
  -H "Content-Type: text/xml" \
  -d '<?xml version="1.0"?><methodCall><methodName>system.client_version</methodName></methodCall>' \
  | grep -o 'string>[0-9.]*'
# Output: 0.16.5
```

### 2. Confirm d.timestamp.load Missing
```bash
$ python3 << 'EOF'
import xmlrpc.client
proxy = xmlrpc.client.ServerProxy("http://localhost:18000/RPC2")
try:
    # This will fail if field doesn't exist
    result = proxy.d.multicall.filtered("", "", "d.hash=", "d.timestamp.load=")
except xmlrpc.client.Fault as e:
    print(f"Error: {e}")
    # Output: Fault -500: 'Command "d.timestamp.load" does not exist.'
EOF
```

### 3. Confirm Workaround (Without d.timestamp.load)
```bash
$ python3 << 'EOF'
import xmlrpc.client
proxy = xmlrpc.client.ServerProxy("http://localhost:18000/RPC2")
# Request WITHOUT d.timestamp.load
results = proxy.d.multicall.filtered("", "", "d.hash=",
    "d.hash=", "d.name=", "d.size_bytes=", "d.completed_bytes=",
    "d.up.rate=", "d.down.rate=", "d.ratio=", "d.state=",
    "d.is_active=", "d.message=", "d.custom1=", "d.peers_accounted=",
    # "d.timestamp.load=" SKIPPED — causes error on 0.16.5
    "d.directory=", "d.hashing="
)
print(f"SUCCESS: Got {len(results)} torrents")
EOF
```

## Solutions

### Short-Term: Compatibility Layer

Modify `silo_client_rt.py` to detect rTorrent version and exclude unsupported fields:

```python
def _get_multicall_fields(proxy):
    """Get field list compatible with rTorrent version."""
    version = proxy.system.client_version()
    major, minor, patch = map(int, version.split('.')[:3])
    
    fields = [
        "d.hash=",           # [0]
        "d.name=",           # [1]
        "d.size_bytes=",     # [2]
        ...
    ]
    
    # d.timestamp.load added in rTorrent 0.17.0
    if (major, minor) >= (0, 17):
        fields.append("d.timestamp.load=")  # [12]
    else:
        # Fallback: use d.creation_date= (available in 0.16.5)
        # OR leave empty (date added = unknown)
        fields.append("")  # Placeholder to keep indices consistent
    
    fields.extend([
        "d.directory=",      # [13]
        "d.hashing=",        # [14]
    ])
    
    return fields
```

Then update `_process_results()` to handle missing/optional fields:

```python
def _process_results(results, proxy=None):
    # created date — optional field
    # Fall back to 0 (unknown) if field doesn't exist
    created = int(item[_F_LOADED]) if len(item) > _F_LOADED and item[_F_LOADED] else 0
```

### Medium-Term: Upgrade rTorrent

Upgrade rTorrent from 0.16.5 to 0.17+ to gain access to `d.timestamp.load`:

```bash
# Current version
$ rtorrent --version
# rtorrent 0.16.5

# Upgrade to 0.17+ (specific steps depend on your package manager)
```

### Long-Term: Version Matrix & Testing

Add CI/testing for multiple rTorrent versions to prevent future regressions:

- Test suite for rTorrent 0.16.x (EOL support)
- Test suite for rTorrent 0.17.x (current stable)
- Test suite for rTorrent 0.18.x (future)

## Recommended Action

**Immediate** (unblock dashboard):
1. Fix `silo_client_rt.py` to gracefully handle missing `d.timestamp.load=`
2. Make "Added" column show "—" (unknown) instead of breaking the entire fetch

**Short-term** (robustness):
1. Add version detection to `silo_client_rt.py`
2. Dynamically exclude unsupported fields based on rTorrent version

**Long-term** (planning):
1. Document rTorrent version compatibility in README
2. Add version check at startup with warning if using unsupported rTorrent

## Impact on Daemon Discovery Feature

This issue is **independent** of the new daemon discovery feature implemented in this session. The discovery feature will correctly locate and start the daemon. However, once started, the daemon will fail due to this rTorrent version incompatibility.

**The daemon discovery feature UNCOVERS this issue** by making the daemon restart attempts visible (they succeed in starting, but fail in executing the fetch).

## Related Issues & Commits

- Commit `f7a5ba4` (2026-03-28): "fix(rt): resolve display issues and switch to d.timestamp.load for Added column"
  - Introduced incompatibility with rTorrent 0.16.5
  - Worked fine on 0.17+ test systems
  - **Not tested on 0.16.5 production system**

---

## Quick Workaround (Until Fixed)

If immediate action is needed:

```bash
# Disable rTorrent client in dashboard (falls back to qBittorrent only)
./bin/silo-dashboard.py --client qbittorrent

# OR use direct API instead of cache
./bin/silo-dashboard.py --rt-direct
```

This allows dashboard to work while the compatibility issue is being fixed.
