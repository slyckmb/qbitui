from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

DASHBOARD_PATH = ROOT / "bin" / "silo-dashboard.py"

spec = importlib.util.spec_from_file_location("qbit_dashboard", DASHBOARD_PATH)
qbit_dashboard = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(qbit_dashboard)


class DummyResult:
    def __init__(self, *, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_fetch_torrents_info_payload_uses_cache_when_available(monkeypatch):
    monkeypatch.setattr(
        qbit_dashboard.subprocess,
        "run",
        lambda *args, **kwargs: DummyResult(returncode=0, stdout='[{"hash":"abc"}]'),
    )

    raw, used_cache, used_direct, error = qbit_dashboard.fetch_torrents_info_payload(
        use_shared_cache=True,
        cache_agent_cmd=Path("/tmp/qbit-cache-agent.py"),
        cache_max_age=15.0,
        cache_wait_fresh=5.0,
        cache_allow_stale=True,
        cache_env={},
        opener=None,
        api_url="http://localhost:9003",
    )

    assert raw == '[{"hash":"abc"}]'
    assert used_cache is True
    assert used_direct is False
    assert error is None


def test_fetch_torrents_info_payload_does_not_fallback_direct_on_cache_failure(monkeypatch):
    monkeypatch.setattr(
        qbit_dashboard.subprocess,
        "run",
        lambda *args, **kwargs: DummyResult(returncode=1, stderr="cache too stale"),
    )

    raw, used_cache, used_direct, error = qbit_dashboard.fetch_torrents_info_payload(
        use_shared_cache=True,
        cache_agent_cmd=Path("/tmp/qbit-cache-agent.py"),
        cache_max_age=15.0,
        cache_wait_fresh=5.0,
        cache_allow_stale=True,
        cache_env={},
        opener=None,
        api_url="http://localhost:9003",
    )

    assert raw == ""
    assert used_cache is False
    assert used_direct is False
    assert "Cache agent failed" in error


def test_fetch_torrents_info_payload_uses_direct_mode_only_when_explicit(monkeypatch):
    monkeypatch.setattr(
        qbit_dashboard,
        "qbit_request",
        lambda opener, api_url, method, path, params=None: '[{"hash":"direct"}]',
    )

    raw, used_cache, used_direct, error = qbit_dashboard.fetch_torrents_info_payload(
        use_shared_cache=False,
        cache_agent_cmd=Path("/tmp/qbit-cache-agent.py"),
        cache_max_age=15.0,
        cache_wait_fresh=5.0,
        cache_allow_stale=True,
        cache_env={},
        opener=None,
        api_url="http://localhost:9003",
    )

    assert raw == '[{"hash":"direct"}]'
    assert used_cache is False
    assert used_direct is True
    assert error is None


def test_build_rows_falls_back_to_tracker_hostname_when_registry_is_empty():
    rows = qbit_dashboard.build_rows(
        [
            {
                "name": "example",
                "state": "stoppedDL",
                "progress": 0.5,
                "size": 1024,
                "tracker": "https://trackerprxy.digitalcore.club/announce/token",
                "hash": "abc123",
            }
        ],
        {},
        {},
    )

    assert rows[0]["tracker"] == "trackerprxy.digitalcore.club"


def test_build_rows_prefers_registry_match_over_hostname_fallback():
    rows = qbit_dashboard.build_rows(
        [
            {
                "name": "example",
                "state": "stoppedDL",
                "progress": 0.5,
                "size": 1024,
                "tracker": "https://tracker.example/announce/token",
                "hash": "abc123",
            }
        ],
        {},
        {r"tracker\.example": "ExampleTracker"},
    )

    assert rows[0]["tracker"] == "ExampleTracker"


def test_find_tracker_registry_resolves_sibling_traktor_from_worktree(monkeypatch, tmp_path):
    silo_root = tmp_path / "tools" / "silo"
    worktree_bin = silo_root / ".agent" / "worktrees" / "chat-123" / "bin"
    worktree_bin.mkdir(parents=True)
    registry = tmp_path / "tools" / "traktor" / "config" / "tracker-registry.yml"
    registry.parent.mkdir(parents=True)
    registry.write_text("trackers: {}\n", encoding="utf-8")

    monkeypatch.delenv("QBIT_TRACKER_REGISTRY_FILE", raising=False)
    monkeypatch.setattr(qbit_dashboard, "__file__", str(worktree_bin / "silo-dashboard.py"))

    assert qbit_dashboard._find_tracker_registry() == registry


def test_status_filter_tracker_issue_matches_rt_tracker_message():
    rows = [
        {
            "name": "seeding with tracker warning",
            "state": "stalledUP",
            "raw": {
                "state": "stalledUP",
                "message": 'Tracker: [Failure reason "Torrent has been deleted."]',
            },
        },
        {
            "name": "fatal incomplete",
            "state": "error",
            "raw": {"state": "error", "message": "Tracker: [Could not connect to server]"},
        },
        {"name": "plain seeding", "state": "stalledUP", "raw": {"state": "stalledUP"}},
    ]

    filtered = qbit_dashboard.apply_filters(
        rows,
        [{"type": "status", "values": ["tracker_issue"], "enabled": True}],
    )

    assert [row["name"] for row in filtered] == ["seeding with tracker warning"]


def test_special_status_terms_are_valid_for_interactive_prompt_filtering():
    assert "tracker_issue" in qbit_dashboard.STATUS_FILTER_TERMS
    assert "ti" in qbit_dashboard.STATUS_FILTER_TERMS
    assert "trk_warn" in qbit_dashboard.STATUS_FILTER_TERMS
    assert "no_working_tracker" in qbit_dashboard.STATUS_FILTER_TERMS
    assert "nt" in qbit_dashboard.STATUS_FILTER_TERMS
    assert "tracker_no_working" in qbit_dashboard.STATUS_FILTER_TERMS


def test_status_filter_tracker_issue_matches_qbit_manage_issue_tag():
    rows = [
        {"name": "qbit tagged", "state": "uploading", "tags": "~issue, tracker", "raw": {"state": "uploading"}},
        {"name": "qbit clean", "state": "uploading", "tags": "tracker", "raw": {"state": "uploading"}},
    ]

    filtered = qbit_dashboard.apply_filters(
        rows,
        [{"type": "status", "values": ["tracker_issue"], "enabled": True}],
    )

    assert [row["name"] for row in filtered] == ["qbit tagged"]


def test_status_filter_ti_alias_matches_tracker_issue():
    rows = [
        {
            "name": "rt warning",
            "state": "stalledUP",
            "raw": {"state": "stalledUP", "message": "Tracker: [Timeout was reached]"},
        },
        {"name": "normal", "state": "stalledUP", "raw": {"state": "stalledUP", "message": ""}},
    ]

    filtered = qbit_dashboard.apply_filters(
        rows,
        [{"type": "status", "values": ["ti"], "enabled": True}],
    )

    assert [row["name"] for row in filtered] == ["rt warning"]


def test_tracker_issue_summary_reports_rt_message():
    row = {
        "name": "rt warning",
        "tracker": "privatehd",
        "state": "stalledUP",
        "raw": {
            "state": "stalledUP",
            "tracker": "https://tracker.example/announce",
            "message": 'Tracker: [Failure reason "Torrent not found"]',
        },
    }

    summary = qbit_dashboard.tracker_issue_summary(row)

    assert summary["type"] == "tracker_issue"
    assert summary["source"] == "rtorrent_message"
    assert summary["tracker"] == "privatehd"
    assert summary["message"] == 'Tracker: [Failure reason "Torrent not found"]'


def test_info_tab_includes_tracker_issue_message():
    row = {
        "name": "rt warning",
        "state": "stalledUP",
        "category": "-",
        "tags": "-",
        "size": "1 GiB",
        "progress": "100%",
        "ratio": "1.0",
        "dlspeed": "0 B/s",
        "upspeed": "0 B/s",
        "eta": "-",
        "hash": "abc",
        "tracker": "privatehd",
        "raw": {"state": "stalledUP", "message": "Tracker: [Timeout was reached]"},
    }

    lines = qbit_dashboard.render_info_lines(row, 100)

    assert "Tracker issue:" in lines
    assert "Type: tracker_issue" in lines
    assert "Source: rtorrent_message" in lines
    assert "Message: Tracker: [Timeout was reached]" in lines


def test_trackers_tab_includes_tracker_issue_banner():
    row = {
        "state": "stalledUP",
        "tracker": "privatehd",
        "raw": {"state": "stalledUP", "message": "Tracker: [Timeout was reached]"},
    }
    trackers = [{"status": "working", "tier": "0", "url": "https://tracker.example/announce"}]

    lines = qbit_dashboard.render_trackers_lines(trackers, 100, row)

    assert lines[:4] == [
        "Tracker issue:",
        "  Type: tracker_issue",
        "  Source: rtorrent_message",
        "  Message: Tracker: [Timeout was reached]",
    ]


def test_status_filter_no_working_tracker_matches_rt_tracker_counts():
    rows = [
        {
            "name": "zero usable trackers",
            "state": "stalledUP",
            "raw": {"state": "stalledUP", "trackers_count": 2, "real_trackers_count": 0},
        },
        {
            "name": "has usable tracker",
            "state": "stalledUP",
            "raw": {"state": "stalledUP", "trackers_count": 2, "real_trackers_count": 1},
        },
    ]

    filtered = qbit_dashboard.apply_filters(
        rows,
        [{"type": "status", "values": ["no_working_tracker"], "enabled": True}],
    )

    assert [row["name"] for row in filtered] == ["zero usable trackers"]


def test_status_filter_nt_alias_matches_no_working_tracker():
    rows = [
        {
            "name": "zero usable trackers",
            "state": "stalledUP",
            "raw": {"state": "stalledUP", "trackers_count": 2, "real_trackers_count": 0},
        },
        {
            "name": "has usable tracker",
            "state": "stalledUP",
            "raw": {"state": "stalledUP", "trackers_count": 2, "real_trackers_count": 1},
        },
    ]

    filtered = qbit_dashboard.apply_filters(
        rows,
        [{"type": "status", "values": ["nt"], "enabled": True}],
    )

    assert [row["name"] for row in filtered] == ["zero usable trackers"]


def test_status_filter_no_working_tracker_ignores_unknown_qbit_tracker_health():
    rows = [
        {
            "name": "qbit unknown tracker health",
            "state": "stalledUP",
            "raw": {"state": "stalledUP", "trackers_count": 2},
        },
        {
            "name": "known zero usable trackers",
            "state": "stalledUP",
            "raw": {"state": "stalledUP", "trackers_count": 2, "real_trackers_count": 0},
        },
    ]

    filtered = qbit_dashboard.apply_filters(
        rows,
        [{"type": "status", "values": ["nt"], "enabled": True}],
    )

    assert [row["name"] for row in filtered] == ["known zero usable trackers"]
