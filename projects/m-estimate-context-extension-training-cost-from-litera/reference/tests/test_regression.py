import sys

sys.path.insert(0, ".")
from ctxcost.estimator import estimate_cost
from ctxcost.rope import simulate_rope_schedule
from ctxcost.evals import compare_strategies


def test_cost_scaling_monotonicity():
    c1 = estimate_cost(1000, 4096, 2048, 1.5)
    c2 = estimate_cost(1000, 8192, 2048, 1.5)
    assert c2 > c1, "cost must increase with target length"


def test_rope_schedule_endpoints():
    sched = simulate_rope_schedule(4, 10000.0, 1000000.0)
    assert len(sched) == 4
    assert abs(sched[0] - 10000.0) < 1e-5
    assert abs(sched[-1] - 1000000.0) < 1e-5


def test_compare_strategies_output():
    res = compare_strategies(5.2, 5.8)
    assert res["preferred"] == "abf"
    assert "perplexity_difference" in res
