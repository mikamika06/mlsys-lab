import sys
sys.path.insert(0, ".")
from vllm_metrics.scraper import parse_exposition
from vllm_metrics.histogram import reconstruct_histogram
from vllm_metrics.reconcile import interpolate_percentile, reconcile_latency

def test_parse_exposition_handles_labels():
    text = 'vllm:request_latency_seconds_bucket{le="0.5",model="llama"} 10.0\n'
    parsed = parse_exposition(text)
    assert "vllm:request_latency_seconds_bucket" in parsed
    labels, val = parsed["vllm:request_latency_seconds_bucket"][0]
    assert labels["le"] == "0.5"
    assert labels["model"] == "llama"
    assert val == 10.0

def test_reconstruct_histogram_sorts_buckets():
    lines = [({"le": "1.0"}, 5.0), ({"le": "0.1"}, 2.0), ({"le": "+Inf"}, 10.0)]
    hist = reconstruct_histogram(lines)
    les = [h[0] for h in hist]
    assert les == [0.1, 1.0, float("inf")]

def test_interpolate_percentile_bounds():
    hist = [(0.1, 2.0), (1.0, 8.0), (float("inf"), 10.0)]
    p90 = interpolate_percentile(hist, 0.90)
    assert p90 > 1.0

def test_reconcile_latency_error_metric():
    latencies = [0.1 * i for i in range(1, 101)]
    hist = [(float(i), float(i * 10)) for i in range(1, 11)] + [(float("inf"), 100.0)]
    res = reconcile_latency(latencies, hist)
    assert "rel_err" in res
    assert res["rel_err"] >= 0.0
