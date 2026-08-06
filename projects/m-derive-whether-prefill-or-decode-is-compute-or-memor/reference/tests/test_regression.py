from prefill.compare import compare_logs

def test_compare_logs_basic():
    """Test log comparison metrics."""
    cfg = {
        "log_entries": [
            {"req_id": 1, "phase": "prefill", "tokens": 100, "time_ms": 10.0},
            {"req_id": 1, "phase": "decode", "tokens": 1, "time_ms": 2.0}
        ],
        "chunked": True
    }
    res = compare_logs(cfg)
    assert res["avg_ttft"] == 10.0
    assert res["avg_itl"] == 2.0
    assert res["chunked"] is True
