import sys
sys.path.insert(0, ".")
from runner.parser import parse_truncated_log
from runner.diagnostics import classify_missing_asset, check_cpu_fallback

def test_parse_truncated_log_basic():
    log = "[INFO] START\nEVENT: init_runtime\nEVENT: load_weights\nFATAL: unexpected EOF"
    res = parse_truncated_log(log)
    assert res["sequence"] == ["init_runtime", "load_weights"]
    assert "FATAL" in res["failure_point"]

def test_classify_missing_manifest():
    ctx = {"message": "Failed to load config.json manifest file"}
    assert classify_missing_asset(ctx) == "manifest_not_found"

def test_detect_cpu_fallback():
    metrics = {"device": "cuda", "cuda_kernels_executed": 0, "cpu_fallback_ops": 15}
    assert check_cpu_fallback(metrics) is True
