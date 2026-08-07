import sys
sys.path.insert(0, ".")
from routing.penalty import compute_overlap, compute_staleness
from routing.cost import routing_cost, select_best_worker


def test_overlap_calculation():
    r = [1, 2, 3, 4, 5]
    w = [1, 2, 3, 99, 100]
    assert compute_overlap(r, w) == 3


def test_staleness_zero_age():
    assert compute_staleness(10, 10, 0.05) == 0.0


def test_routing_cost_basic():
    c = routing_cost(10, 20, 0.1, 1.0, 5.0)
    assert c > 0.0


def test_select_best_worker_prefers_better_match():
    req = [10, 20, 30, 40]
    workers = [
        {"worker_id": "w1", "cached_tokens": [10, 20, 99, 99], "last_access_tick": 0, "decay_factor": 0.0},
        {"worker_id": "w2", "cached_tokens": [10, 20, 30, 40], "last_access_tick": 0, "decay_factor": 0.0}
    ]
    best = select_best_worker(req, workers, 1.0, 5.0, 10)
    assert best == "w2"
