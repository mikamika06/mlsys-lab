import sys

sys.path.insert(0, ".")
from compress.api import pareto_frontier


def test_pareto_filters_dominated():
    pts = [(10, 90), (15, 85), (20, 95)]
    front = pareto_frontier(pts)
    assert (15, 85) not in front
    assert (10, 90) in front
    assert (20, 95) in front
    assert len(front) == 2


def test_pareto_keeps_all_optimal_and_sorts():
    pts = [(30, 95), (10, 80), (20, 90)]
    front = pareto_frontier(pts)
    assert len(front) == 3
    assert front[0] == (10, 80)
    assert front[1] == (20, 90)
    assert front[2] == (30, 95)
