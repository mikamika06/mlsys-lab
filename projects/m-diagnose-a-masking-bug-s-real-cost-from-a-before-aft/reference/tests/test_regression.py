import sys
sys.path.insert(0, ".")
from diagnose.metrics import parse_metrics
from diagnose.diff import compute_diff
from diagnose.cost import evaluate_cost

BASE_CSV = """
mcu_inst_executed,1000000.0
stall_mio_throttle,200000.0
registers_per_thread,32.0
sm_efficiency,85.0
"""

MASK_CSV = """
mcu_inst_executed,1200000.0
stall_mio_throttle,250000.0
registers_per_thread,40.0
sm_efficiency,70.0
"""


def test_parse_metrics_valid():
    parsed = parse_metrics(BASE_CSV)
    assert "mcu_inst_executed" in parsed
    assert parsed["mcu_inst_executed"] == 1000000.0


def test_diff_ratio_calculation():
    b = parse_metrics(BASE_CSV)
    m = parse_metrics(MASK_CSV)
    d = compute_diff(b, m)
    assert d["mcu_inst_executed"]["ratio"] == 1.2


def test_cost_evaluation_catches_regression():
    b = parse_metrics(BASE_CSV)
    m = parse_metrics(MASK_CSV)
    d = compute_diff(b, m)
    res = evaluate_cost(d)
    assert res["regression_detected"] is True
