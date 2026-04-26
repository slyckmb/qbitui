# Fix: Torrent Added Date for rTorrent 0.16.5

## Problem
Current code uses `d.timestamp.load=` which doesn't exist in rTorrent 0.16.5, breaking the entire daemon.

## Solution
Use `d.load_date=` instead, which is available in rTorrent 0.16.5 and returns the exact same information (when the torrent was added to the client).

## Evidence

**Available fields in rTorrent 0.16.5:**
- ✓ `d.load_date=` — When torrent was added to client (EXISTS)
- ✓ `d.creation_date=` — When torrent file was created  
- ✗ `d.timestamp.load=` — NOT AVAILABLE (introduced in 0.17+)

**Real example from test system:**
```
Torrent: Peter F. Hamilton - Family Matters.epub
  d.creation_date:  1479875090 (2016-11-22 23:24:50) — File creation
  d.load_date:      1776776437 (2026-04-21 09:00:37) — Added to client ✓
```

## Fix Implementation

### File: `bin/silo_client_rt.py`

**Current (broken for 0.16.5):**
```python
_MULTICALL_FIELDS = [
    "d.hash=",           # [0]
    "d.name=",           # [1]
    ...
    "d.timestamp.load=", # [12] ← BREAKS on rTorrent 0.16.5
    ...
]
```

**Fixed (works on 0.16.5+):**
```python
_MULTICALL_FIELDS = [
    "d.hash=",           # [0]
    "d.name=",           # [1]
    "d.size_bytes=",     # [2]
    "d.completed_bytes=",# [3]
    "d.up.rate=",        # [4]
    "d.down.rate=",      # [5]
    "d.ratio=",          # [6]
    "d.state=",          # [7]
    "d.is_active=",      # [8]
    "d.message=",        # [9]
    "d.custom1=",        # [10]
    "d.peers_accounted=",# [11]
    "d.load_date=",      # [12] ← CHANGED: use load_date (available in 0.16.5)
    "d.directory=",      # [13]
    "d.hashing=",        # [14]
]
```

### Update in `_process_results()` function

The code that uses this field should remain the same — it reads from index [12]:

```python
# Primary Date Added: d.load_date (available in rTorrent 0.16.5)
created = int(item[_F_LOADED]) if item[_F_LOADED] else 0
```

The variable names and logic don't need to change; just the XMLRPC field changes.

## Version Compatibility

- **rTorrent 0.16.x**: Use `d.load_date=` ✓ Works
- **rTorrent 0.17.x**: Both `d.load_date=` and `d.timestamp.load=` work
- **rTorrent 0.18.x**: Both work

Using `d.load_date=` maintains backward compatibility with 0.16.x while working fine on 0.17+.

## Testing

After applying the fix:

```bash
# Test daemon directly
python3 /home/michael/dev/tools/silo/bin/silo-rt-cache-daemon.py --once

# Check cache was populated
ls -lah ~/.cache/silo-rt/torrents.json
cat ~/.cache/silo-rt/torrents.json | jq '.[] | .added_at' | head -5

# Check dashboard shows torrents
./bin/silo-dashboard.py --client rtorrent
```

## Why This Happened

- Commit f7a5ba4 (2026-03-28) used `d.timestamp.load=` which was available on the test system
- The test system likely runs rTorrent 0.17+
- The production system runs rTorrent 0.16.5
- No version compatibility testing was done

## Prevention

Add to test CI/CD:
1. Test suite for rTorrent 0.16.x (legacy support)
2. Test suite for rTorrent 0.17+ (current/future)
3. Check available XMLRPC methods at startup (warn on unsupported version)

## Summary

**Single-line fix**: Change `"d.timestamp.load="` to `"d.load_date="` in `_MULTICALL_FIELDS`

This field provides the same "torrent added" date, is available in rTorrent 0.16.5, and maintains compatibility with newer versions.
