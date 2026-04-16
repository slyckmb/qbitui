# Handoff Prompt — 2026-04-16

Use this as bootstrap for next-agent or next-session work:

```text
Work in:
- /home/michael/dev/tools/silo
- branch main (or cr/* for feature work)

Read first:
1. docs/agent-comms/sitrep-2026-04-16.md
2. docs/TRACKER.md (sections P2, P3)
3. git log --oneline -10

Current known-good state:
- qB and RT TRK columns resolve tracker keys from traktor registry URL patterns
- RT completed tracker-message items no longer appear in error scope
- Local traktor registry includes torrentday alternate hosts; live RT verifies td.jumbohostpro.eu
- All three clients (qBit, rTorrent, SABnzbd) use unified schema in dashboard
- Scope filtering (w/u/v/e/g) works for qBit; broken for rTorrent and SABnzbd (P2-B/C)

Next priority work:
1. P2-B/C: Scope filter fix (unified state key instead of raw.state)
   - File: bin/silo-dashboard.py, rows_to_render filter block
   - Impact: widest UX fix
2. P2-A: MediaInfo cache miss (async placeholder)
   - File: bin/silo-dashboard.py, get_mediainfo_summary_cached()

Validation shortcuts:
- python3 -m py_compile bin/silo_client_rt.py bin/silo-dashboard.py tests/*.py
- PYTHONPATH=bin python3 -c "from silo_client_rt import *; from silo_client_sab import *; print('OK')"
- Inspect live dashboard with: python3 bin/silo-dashboard.py

Do not re-open:
- e66089c fix(rt): resolve tracker labels from registry
- 2844290 fix(rt): keep completed tracker warnings out of errors
  (unless live behavior contradicts the expected state)
```
