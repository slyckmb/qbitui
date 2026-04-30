from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import qb_cache_lib  # noqa: E402


def test_default_qb_cache_base_preserves_hashall_cache_path():
    assert qb_cache_lib.DEFAULT_QB_CACHE_BASE == Path.home() / ".cache" / "hashall-qb"
