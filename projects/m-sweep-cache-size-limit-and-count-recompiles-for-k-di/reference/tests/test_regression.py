import sys
sys.path.insert(0, ".")
from compiler_diag.storm import simulate_recompile_storm

def test_fallback_detected_on_cache_exhaustion():
    shapes = [(1, 10), (2, 10), (3, 10), (4, 10), (5, 10)]
    limit = 3
    res = simulate_recompile_storm(shapes, limit)
    assert res["fallback_step"] == 3
    assert res["total_recompiles"] == 3
    assert res["total_fallbacks"] == 2
    assert res["history"][3]["status"] == "eager_fallback"

def test_no_fallback_when_within_limit():
    shapes = [(1, 10), (2, 10)]
    limit = 5
    res = simulate_recompile_storm(shapes, limit)
    assert res["fallback_step"] is None
    assert res["total_recompiles"] == 2
    assert res["total_fallbacks"] == 0
