import sys

sys.path.insert(0, ".")
from ringbal.assign import zigzag_assignment
from ringbal.metrics import workload_imbalance


def test_zigzag_is_perfectly_balanced():
    assignment = zigzag_assignment(64, 8)
    stats = workload_imbalance(assignment)
    assert abs(stats["rel_err"]) < 1e-7, f"imbalance rel_err is {stats['rel_err']}, expected 0.0"
