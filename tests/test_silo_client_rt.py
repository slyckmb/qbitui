from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "bin" / "silo_client_rt.py"

spec = importlib.util.spec_from_file_location("silo_client_rt", CLIENT_PATH)
silo_client_rt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(silo_client_rt)


def test_normalize_state_keeps_completed_tracker_rejections_out_of_error():
    state = silo_client_rt._normalize_state(
        state=1,
        is_active=1,
        message='Tracker: [Failure reason "Torrent has been deleted."]',
        size_bytes=1024,
        completed_bytes=1024,
        dl_rate=0,
        up_rate=0,
        hashing=0,
    )

    assert state == "stalledUP"


def test_normalize_state_keeps_paused_completed_items_out_of_error():
    state = silo_client_rt._normalize_state(
        state=1,
        is_active=0,
        message='Tracker: [Failure reason "Passkey does not exist! Please Re-download the .torrent"]',
        size_bytes=1024,
        completed_bytes=1024,
        dl_rate=0,
        up_rate=0,
        hashing=0,
    )

    assert state == "pausedUP"


def test_normalize_state_keeps_stopped_incomplete_items_stopped():
    state = silo_client_rt._normalize_state(
        state=0,
        is_active=0,
        message="Tracker: [Timeout was reached]",
        size_bytes=1024,
        completed_bytes=512,
        dl_rate=0,
        up_rate=0,
        hashing=0,
    )

    assert state == "stoppedDL"


def test_normalize_state_marks_active_incomplete_message_as_error():
    state = silo_client_rt._normalize_state(
        state=1,
        is_active=1,
        message="Tracker: [Could not connect to server]",
        size_bytes=1024,
        completed_bytes=512,
        dl_rate=0,
        up_rate=0,
        hashing=0,
    )

    assert state == "error"


def test_tracker_label_from_url_prefers_registry_pattern(monkeypatch, tmp_path):
    registry = tmp_path / "tracker-registry.yml"
    registry.write_text(
        "trackers:\n"
        "  torrentday:\n"
        "    qbitmanage:\n"
        "      tracker_url_pattern: 'torrentday|td\\.jumbohostpro\\.eu|sync\\.td-peers\\.com'\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(silo_client_rt, "_TRACKER_URL_PATTERN_MAP", None)
    monkeypatch.setattr(silo_client_rt, "_find_tracker_registry", lambda: registry)
    monkeypatch.setitem(
        sys.modules,
        "yaml",
        SimpleNamespace(
            safe_load=lambda _text: {
                "trackers": {
                    "torrentday": {
                        "qbitmanage": {
                            "tracker_url_pattern": r"torrentday|td\.jumbohostpro\.eu|sync\.td-peers\.com"
                        }
                    }
                }
            }
        ),
    )

    label = silo_client_rt._tracker_label_from_url(
        "http://td.jumbohostpro.eu/tNtOztQuQIzYElZyDuuJIeLJW8chntAH/announce"
    )

    assert label == "torrentday"


def test_tracker_label_from_url_falls_back_to_hostname(monkeypatch):
    monkeypatch.setattr(silo_client_rt, "_TRACKER_URL_PATTERN_MAP", {})

    label = silo_client_rt._tracker_label_from_url("https://example.tracker.invalid/announce")

    assert label == "example.tracker.invalid"


def test_find_tracker_registry_resolves_sibling_traktor_from_worktree(monkeypatch, tmp_path):
    silo_root = tmp_path / "tools" / "silo"
    worktree_bin = silo_root / ".agent" / "worktrees" / "chat-123" / "bin"
    worktree_bin.mkdir(parents=True)
    registry = tmp_path / "tools" / "traktor" / "config" / "tracker-registry.yml"
    registry.parent.mkdir(parents=True)
    registry.write_text("trackers: {}\n", encoding="utf-8")

    monkeypatch.delenv("QBIT_TRACKER_REGISTRY_FILE", raising=False)
    monkeypatch.setattr(silo_client_rt, "__file__", str(worktree_bin / "silo_client_rt.py"))

    assert silo_client_rt._find_tracker_registry() == registry
