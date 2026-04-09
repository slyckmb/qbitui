from pathlib import Path
import importlib.util


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
