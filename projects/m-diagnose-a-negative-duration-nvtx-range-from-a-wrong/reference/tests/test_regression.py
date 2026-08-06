from nvdiag.analyzer import diagnose_range


def test_nvtx_thread_consistency():
    events = [
        {"pid": 1, "tid": 10, "ts": 100, "name": "Step", "ph": "B"},
        {"pid": 1, "tid": 20, "ts": 150, "name": "Step", "ph": "E"}
    ]
    res = diagnose_range(events)
    assert res is not None
    assert "wrong_tid" in res
    assert res["wrong_tid"] != res["correct_tid"]
    assert "duration_clamped" not in res
