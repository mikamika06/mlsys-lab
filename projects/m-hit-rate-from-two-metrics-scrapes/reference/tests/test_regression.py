import sys

sys.path.insert(0, ".")
from kvmetric.calc import hit_rate, build_promql


def test_hit_rate_bounds():
    prev_s = {"kv_cache_hits_total": 10, "kv_cache_misses_total": 10}
    curr_s = {"kv_cache_hits_total": 30, "kv_cache_misses_total": 20}
    hr = hit_rate(prev_s, curr_s)
    assert 0.0 <= hr <= 1.0


def test_promql_structure():
    q = build_promql("kv_cache_hits_total", "counter", "5m")
    assert "rate" in q
    assert "sum" in q
    assert "by (instance)" in q
