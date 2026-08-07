import sys
sys.path.insert(0, ".")
from goodput.metrics import compute_goodput, compute_e2el_gap, compute_itl_and_tpot


def test_goodput_boundaries():
    traces = [
        {"ttft": 100.0, "tpot": 20.0, "num_tokens": 5, "timestamps": [0, 100, 120, 140, 160]},
        {"ttft": 250.0, "tpot": 20.0, "num_tokens": 5, "timestamps": [0, 250, 270, 290, 310]}
    ]
    gp = compute_goodput(traces, 200.0, 30.0)
    assert gp == 0.5


def test_e2el_gap_non_negative():
    traces = [
        {"ttft": 100.0, "tpot": 20.0, "num_tokens": 3, "timestamps": [0, 100, 150, 170]}
    ]
    gap = compute_e2el_gap(traces)
    assert gap >= 0.0


def test_itl_divergence():
    ts = [0.0, 100.0, 120.0, 400.0, 420.0]
    tpot, max_itl = compute_itl_and_tpot(ts)
    assert max_itl > tpot
