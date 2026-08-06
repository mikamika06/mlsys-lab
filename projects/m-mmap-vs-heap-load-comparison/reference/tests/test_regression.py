import sys
sys.path.insert(0, ".")
from weightloader.dedup import compute_dedup_savings
from weightloader.loader import simulate_load


def test_dedup_savings_non_negative():
    tensors = [
        {"name": "a", "size": 1024, "hash": "abc"},
        {"name": "b", "size": 1024, "hash": "abc"},
        {"name": "c", "size": 2048, "hash": "def"}
    ]
    res = compute_dedup_savings(tensors)
    assert res["savings"] >= 0, "savings cannot be negative"
    assert res["unique_bytes"] == 3072


def test_loader_peak_rss_positive():
    tensors = [{"name": "a", "size": 4096, "hash": "abc"}]
    res = simulate_load(tensors, "heap")
    assert res["peak_rss"] > 0, "peak rss must be positive"


def test_attribution_keys():
    from weightloader.attribution import attribute_regression
    b = {"peak_rss": 100}
    n = {"peak_rss": 150}
    res = attribute_regression(b, n)
    assert "peak_rss" in res
