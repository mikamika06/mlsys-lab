import sys
sys.path.insert(0, ".")
from runners.baseline import setup_baseline
from runners.compare import run_comparison
from runners.metrics import compute_metrics

CONFIG = {"parameters": 7, "quantization": "Q4_0"}
WORKLOAD = [100, 200, 300]

def test_baseline_consistency():
    b = setup_baseline(CONFIG)
    assert b["name"] == "llamafile"
    assert b["ready"] is True

def test_runner_count():
    runners = [
        setup_baseline(CONFIG),
        {"name": "runner_b", "scale": 1.01},
        {"name": "runner_c", "scale": 0.99}
    ]
    res = run_comparison(runners, WORKLOAD)
    assert len(res) == 3

def test_relative_error_bound():
    b = setup_baseline(CONFIG)
    runners = [
        {"name": "llamafile", "scale": 1.0},
        {"name": "runner_b", "scale": 1.02}
    ]
    res = run_comparison(runners, WORKLOAD)
    metrics = compute_metrics({"tokens": WORKLOAD}, res)
    for m in metrics:
        assert m["max_rel_err"] < 0.1
