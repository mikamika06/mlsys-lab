import sys
sys.path.insert(0, ".")
from servingmetrics.analysis import compute_goodput, analyze_preemption_gap, compute_itl_and_tpot

def test_goodput_basic():
    results = [
        {"request_id": 1, "status": "success", "arrival_time": 0.0, "first_token_time": 0.1, "token_timestamps": [0.1, 0.15, 0.2]},
        {"request_id": 2, "status": "success", "arrival_time": 0.0, "first_token_time": 0.3, "token_timestamps": [0.3, 0.4, 0.5]}
    ]
    ratio, count, acc = compute_goodput(results, 200.0, 30.0)
    assert count == 1.0

def test_preemption_gap_calculation():
    trace = {
        "arrival_time": 0.0,
        "first_token_time": 0.05,
        "preemption_time": 0.5,
        "token_timestamps": [0.05, 0.1, 0.7, 0.75]
    }
    res = analyze_preemption_gap(trace)
    assert "actual_gap" in res

def test_itl_tpot_divergence():
    timestamps = [0.0, 0.1, 0.12, 0.13, 0.5]
    res = compute_itl_and_tpot(timestamps)
    assert res["divergence"] > 0.0
