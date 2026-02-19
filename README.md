# qbitui

Interactive terminal dashboard for [qBittorrent](https://www.qbittorrent.org/). Navigate, filter, and manage your torrents without leaving the terminal.

## Features

- **Zero dependencies** — pure Python 3 stdlib (optional `pyyaml` for config files)
- **Fast TUI** — paged list view with minimal redraws, tmux-friendly
- **Filter & sort** — live filter prompt, tab-based category views
- **Selection mode** — multi-select with bulk operations
- **MediaInfo** — optional background mediainfo cache with `--mediainfo`
- **Hotkey-driven** — no arrow keys required; works cleanly over SSH/mosh

## Requirements

- Python 3.9+
- qBittorrent with Web UI enabled
- `mediainfo` (optional, for media metadata column)

## Usage

```bash
# Basic — connects to localhost:8080
./bin/qbit-dashboard.py

# Custom host/port
./bin/qbit-dashboard.py --host 10.0.0.10 --port 9003

# With credentials
./bin/qbit-dashboard.py --username admin --password secret

# Enable mediainfo column
./bin/qbit-dashboard.py --mediainfo
```

### Config file (optional)

Place a `config/qbit-dashboard.yml` (or set `QBITTORRENT_CONFIG_FILE`) with:

```yaml
qbittorrent:
  api_url: http://localhost:9003
  username: admin
  password_file: /path/to/password.env
```

## Keymap

| Key | Action |
|-----|--------|
| `,` / `.` | Page prev / next |
| `'` / `/` | Cursor up / down |
| `l` | Filter prompt |
| `t` | Toggle tags column |
| `Tab` | Cycle category tabs |
| `Space` / `Enter` | Toggle selection |
| `Esc` | Clear selection / exit tab |
| `z` | Reset to default view |
| `q` | Quit |

## Install

```bash
git clone https://github.com/<you>/qbitui
cd qbitui
# Optional: symlink into PATH
ln -s "$PWD/bin/qbit-dashboard.py" ~/.local/bin/qbitui
```

## Status

Actively used. Arrow key support intentionally removed (tmux ESC sequence conflicts). A non-blocking input loop and curses-style buffered screen are planned improvements.
