import sys
sys.path.insert(0, ".")
from vllmbench.metrics import compute_percentiles
from vllmbench.arrival import generate_poisson_arrivals
from vllmbench.loop import evaluate_loops


def test_percentiles_basic():
    data = [{"ttft": float(i), "tpot": 1.0, "itl": 0.5, "e2el": float(i * 2)} for i in range(100)]
    res = compute_percentiles(data)
    assert res["ttft"]["p50"] == 49.5


def test_poisson_count():
    arrivals = generate_poisson_arrivals(10.0, 5.0, seed=123)
    assert len(arrivals) > 0


def test_loop_evaluation():
    res = evaluate_loops(4, 0.1, 1.0)
    assert res["saturation_divergence"] is True
