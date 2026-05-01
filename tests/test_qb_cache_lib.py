from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))

import qb_cache_lib  # noqa: E402


def test_default_qb_cache_base_is_silo_owned_path():
    assert qb_cache_lib.DEFAULT_QB_CACHE_BASE == Path.home() / ".cache" / "silo-qb"


def test_legacy_qb_cache_base_preserves_read_fallback_path():
    assert qb_cache_lib.LEGACY_QB_CACHE_BASE == Path.home() / ".cache" / "hashall-qb"


def test_legacy_cache_pair_only_applies_to_default_silo_cache_file():
    cache_file = qb_cache_lib.DEFAULT_QB_CACHE_BASE / "torrents-info.json"
    meta_file = qb_cache_lib.DEFAULT_QB_CACHE_BASE / "torrents-info.meta.json"

    legacy_cache, legacy_meta = qb_cache_lib._legacy_cache_pair(cache_file, meta_file)

    assert legacy_cache == qb_cache_lib.LEGACY_QB_CACHE_BASE / "torrents-info.json"
    assert legacy_meta == qb_cache_lib.LEGACY_QB_CACHE_BASE / "torrents-info.meta.json"
    custom = Path("/tmp/custom-qb.json")
    assert qb_cache_lib._legacy_cache_pair(custom, custom.with_suffix(".meta.json"))[0] == custom
