from memdiag.leak import diagnose_leak

def test_diagnose_leak_monotonic():
    snaps = [{"step": 0, "active_bytes": 100}, {"step": 1, "active_bytes": 200}]
    assert diagnose_leak(snaps) is True
