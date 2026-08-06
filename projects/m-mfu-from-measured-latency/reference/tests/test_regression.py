import sys
sys.path.insert(0, ".")
from mfu.accounting import compute_mfu, compute_tflops
from mfu.audit import audit_speedup


def test_mfu_bounds():
    mfu = compute_mfu(10.0, 2e12, 312.0)
    assert 0.0 < mfu <= 1.0


def test_tflops_calculation():
    tf = compute_tflops(1e12, 5.0)
    assert abs(tf - 200.0) < 1e-5


def test_audit_consistency():
    res = audit_speedup(100.0, 50.0, 2.0)
    assert res["is_consistent"] is True
    res_bad = audit_speedup(100.0, 50.0, 3.5)
    assert res_bad["is_consistent"] is False
