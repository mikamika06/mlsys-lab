from crossover.audit import audit_benchmark

def test_audit_flags_unfair():
    bad_bench = {"name": "test", "raw_tps": 100.0, "gpus": 1, "precision": "INT4", "warmup": False, "requests": 1}
    issues = audit_benchmark(bad_bench)
    assert len(issues) > 0
    assert "no_warmup" in issues
    assert "single_request" in issues
