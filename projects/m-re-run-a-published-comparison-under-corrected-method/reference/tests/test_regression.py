import sys
sys.path.insert(0, ".")
from runner_audit.audit import detect_context_mismatch, compute_required_repeats


def test_detect_context_mismatch_identical():
    run_a = {"context_length": 2048}
    run_b = {"context_length": 2048}
    assert detect_context_mismatch(run_a, run_b) is False


def test_detect_context_mismatch_different():
    run_a = {"context_length": 2048}
    run_b = {"context_length": 4096}
    assert detect_context_mismatch(run_a, run_b) is True


def test_compute_required_repeats_positive():
    latencies = [10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.1, 9.9, 10.0, 10.1]
    n = compute_required_repeats(latencies, target_rel_error=0.05)
    assert isinstance(n, int)
    assert n > 0
