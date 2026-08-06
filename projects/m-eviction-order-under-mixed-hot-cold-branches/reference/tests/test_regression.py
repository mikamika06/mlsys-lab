import sys

sys.path.insert(0, ".")
from evict.policy import compute_scores
from evict.tree import eviction_order
from evict.scheduler import simulate_reclaim

CONFIG = {
    "nodes": {
        0: {"parent": None, "children": [1, 2], "access_count": 100, "last_access": 10, "is_hot": True},
        1: {"parent": 0, "children": [3], "access_count": 80, "last_access": 9, "is_hot": True},
        2: {"parent": 0, "children": [], "access_count": 5, "last_access": 1, "is_hot": False},
        3: {"parent": 1, "children": [], "access_count": 70, "last_access": 8, "is_hot": True},
    },
    "capacity": 2
}


def test_scores_monotonicity():
    scores = compute_scores(CONFIG)
    assert scores[0] > scores[2]


def test_eviction_order_sorts_correctly():
    order = eviction_order(CONFIG)
    scores = compute_scores(CONFIG)
    for i in range(len(order) - 1):
        assert scores[order[i]] <= scores[order[i+1]]


def test_reclaim_respects_capacity():
    reclaimed = simulate_reclaim(CONFIG)
    total = len(CONFIG["nodes"])
    cap = CONFIG["capacity"]
    assert len(reclaimed) == max(0, total - cap)
