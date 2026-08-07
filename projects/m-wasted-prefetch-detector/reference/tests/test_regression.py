import sys
sys.path.insert(0, ".")
from prefetch.histogram import compute_reuse_histogram
from prefetch.detector import detect_wasted
from prefetch.ttft import model_ttft_savings

def test_histogram_length():
    trace = [{"time": 0, "block_id": 1, "is_prefetch": False}]
    h = compute_reuse_histogram(trace, 10)
    assert len(h) == 11

def test_detect_wasted_identifies_prefetch():
    trace = [
        {"time": 0, "block_id": 10, "is_prefetch": True},
        {"time": 1, "block_id": 20, "is_prefetch": False},
        {"time": 2, "block_id": 30, "is_prefetch": False}
    ]
    w = detect_wasted(trace, 1)
    assert len(w) > 0

def test_ttft_savings_behavior():
    trace = [{"time": 0, "block_id": 1, "is_prefetch": False}]
    res = model_ttft_savings(trace, [5])
    assert 5 in res
    assert res[5] >= 0.0
