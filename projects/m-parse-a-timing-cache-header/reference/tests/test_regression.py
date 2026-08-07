import sys

sys.path.insert(0, ".")
from trtcache.parser import is_reusable


def test_rejects_ablated_tactic_sources():
    cache_header = {
        "version": 8600,
        "sm_major": 8,
        "sm_minor": 9,
        "tactic_sources": 7,
        "opt_level": 3
    }
    builder_config = {
        "version": 8600,
        "sm_major": 8,
        "sm_minor": 9,
        "tactic_sources": 3,
        "opt_level": 3
    }

    reusable = is_reusable(cache_header, builder_config)
    assert not reusable, "Should reject cache containing ablated tactic sources"
