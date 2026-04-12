# Critical Context — 2026-04-12

## RT tracker label path

1. `bin/silo_client_rt.py` fetches RT tracker URLs via `t.multicall`.
2. RT rows now resolve tracker URLs through traktor `qbitmanage.tracker_url_pattern`.
3. Fallback remains hostname-only when no registry pattern matches.
4. RT cache daemon serializes already-resolved RT rows; dashboard does not need to remap RT rows itself.

## Registry dependency

- Tracker-key resolution depends on local file:
  - `/home/michael/dev/tools/traktor/config/tracker-registry.yml`
- Worktree-safe registry discovery was added in both dashboard/qB path and RT path.
- If a tracker shows as a hostname again, first check local registry content before changing silo code.

## RT error split

- Reference behavior came from:
  - `/mnt/config/docker/gluetun_qbit/rtorrent_vpn/bin/rt-watch.sh --dashboard`
- Desired rule:
  - complete item + `d.message` => tracker warning only, keep seeding/paused/stopped state
  - active incomplete item + `d.message` => `error`

## Tests present

- `tests/test_silo_dashboard_cache.py`
- `tests/test_silo_client_rt.py`

## Verification available here

- `python3 -m py_compile ...` works
- `pytest` is not installed in this environment
- live qB / RT checks were done with one-off `python3` scripts
