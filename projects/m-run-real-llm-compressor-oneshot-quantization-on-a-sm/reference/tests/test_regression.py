import sys
sys.path.insert(0, ".")
from compressor_workflow.metrics import measure_metrics
from compressor_workflow.quantize import execute_oneshot

def test_compression_ratio_positive():
    res = measure_metrics(1000, 500, "dummy", [])
    assert res["compression_ratio"] > 0.0

def test_perplexity_finite():
    res = measure_metrics(1000, 500, "dummy", [])
    assert res["perplexity"] > 0.0

def test_oneshot_returns_dict():
    res = execute_oneshot("dummy")
    assert isinstance(res, dict)
