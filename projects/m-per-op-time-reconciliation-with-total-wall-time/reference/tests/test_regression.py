import sys
sys.path.insert(0, ".")
from reconcile.profiler import reconcile_profile_times, calculate_overhead_ratio


def test_reconciliation_invariant():
    sample_report = {
        "total_wall_time_us": 10000.0,
        "ops": [
            {"name": "op1", "real_time_us": 4000.0},
            {"name": "op2", "real_time_us": 5800.0}
        ]
    }
    res = reconcile_profile_times(sample_report)
    assert res["reconciled"] is True
    assert abs(res["overhead_ratio"] - 0.02) < 1e-6


def test_overhead_flagging():
    unreconciled_report = {
        "total_wall_time_us": 10000.0,
        "ops": [
            {"name": "op1", "real_time_us": 2000.0}
        ]
    }
    res = reconcile_profile_times(unreconciled_report)
    assert res["reconciled"] is False
    assert res["overhead_ratio"] > 0.1
