# Hashall Agent: Joint Comms File — Monitor & Participate

## What this file is

A shared coordination log used by four agents (silo, hashall, docker, traktor) to
track progress, deconflict work, and exchange requests across repos. All agents
append to it; none rewrites prior entries.

**Canonical path:**
```
/tmp/cr-docker-20260329-175737-codex-rt-hardening-shared-comms-2026-04-02.md
```

---

## How to monitor it

Use `inotifywait` so you get notified on every write:

```bash
inotifywait -m -e close_write \
  /tmp/cr-docker-20260329-175737-codex-rt-hardening-shared-comms-2026-04-02.md
```

On each event, read the tail of the file to find new entries (look for new
`### YYYY-MM-DD` headings after your last-seen timestamp).

---

## How to post an update

Append — never edit prior content:

```bash
cat >> /tmp/cr-docker-20260329-175737-codex-rt-hardening-shared-comms-2026-04-02.md << 'UPDATE'

### YYYY-MM-DD HH:MM EDT - hashall (<agent-id>)
- what changed / what you found
- what remains
- blockers / asks (addressed to specific agents if needed)
UPDATE
```

Use real timestamps. Include your agent session ID in the header line.

---

## Current state (as of 2026-04-03 ~17:45 EDT)

Hardening is largely complete. Active lanes are:

| Lane | Owner | Status |
|------|-------|--------|
| Pool migration converged | Hashall | ✅ /pool/data torrent rows = 0 |
| /pool/data payload residue | Hashall | 🔄 35 rows remaining |
| RT error reduction | Docker | 🔄 error=2514, timeout=1926 |
| qbit_only=119 classification | Docker | 🔄 mostly dead-tracker residue |
| Hitchhiker splitting | Docker | 🔄 91 shared groups |
| Savepath convergence | Docker | 🔄 379 remaining |
| rt_only=2 sonarr rows | Docker | ⏳ blocked on legacy paths |

---

## Hashall's standing policy (already established)

**Cache-first for reads:**
- `hashall rt state-audit` (default): reads `~/.cache/silo-rt/torrents.json`
- `hashall rt state-audit --live`: allowed for explicit diagnostics only
- All report/audit paths: cache-backed, fail closed on stale cache

**Direct RT XMLRPC allowed only for:**
- `rt repoint`, `rt recheck`, `rt session-reset`, `rt repair-apply`
- `rt state-audit --live`

**qB:**
- Still a hard dependency for `payload sync` and `rehome apply`
- Do not assume qB can be removed

---

## Silo cache contract (stable)

```
Cache file:  ~/.cache/silo-rt/torrents.json       # array of display dicts
Meta file:   ~/.cache/silo-rt/torrents.meta.json
```

Key meta fields: `active_transport`, `using_fallback`, `primary_failures`,
`cache_age_s`, `last_error`, `items`, `source`

Key `raw` fields per torrent: `tracker`, `trackers`, `trackers_http`,
`trackers_count`, `real_trackers_count`, `complete`, `hashing`

**Note:** `status=working` in a tracker row reflects the last RT session state,
not current reachability. Stale for stopped torrents — do not treat as a live probe.

---

## Deconfliction rules

- Post a note before mutating paths that another agent may be reading.
- Name the specific hashes/families if locking a lane.
- Release the hold explicitly when you're done.
- Do not mutate Docker's hitchhiker-split families until Docker posts
  "affected hashes/roots — splitting complete."
- Docker does not touch Hashall's /pool/data cleanup lane.

---

## Agents active on this file

| Agent | Repo | Session ID |
|-------|------|------------|
| silo | `/home/michael/dev/tools/silo` | cr-silo-20260326-113125-claude |
| hashall | `/home/michael/dev/work/hashall` | cr-hashall-20260319-130301-codex |
| docker | `/mnt/config/docker` | cr-docker-20260329-175737-codex |
| traktor | (traktor repo) | traktor-20260403-142808-claude |
