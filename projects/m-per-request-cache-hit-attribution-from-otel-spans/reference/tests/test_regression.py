import sys

sys.path.insert(0, ".")
from otel_cache.attribution import extract_attribution
from otel_cache.metrics import compute_efficiency


def test_extraction_basic():
    spans = [{"attributes": {"request.id": "r1", "kv.cache.hit_tokens": 10, "kv.cache.total_tokens": 100}}]
    res = extract_attribution(spans)
    assert "r1" in res
    assert res["r1"]["hit_tokens"] == 10


def test_efficiency_computation():
    attr = {"r1": {"hit_tokens": 50, "total_tokens": 100}}
    assert compute_efficiency(attr) == 0.5
