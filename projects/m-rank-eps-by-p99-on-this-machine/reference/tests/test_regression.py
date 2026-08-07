import sys

sys.path.insert(0, ".")
from ep_selector.fragmentation import compute_fragmentation_cost
from ep_selector.policy import select_best_backend
from ep_selector.ranking import rank_execution_providers


def test_ranking_sorts_by_p99():
    latencies = {"ort-cpu": [10.0, 12.0, 100.0], "ort-cuda": [2.0, 3.0, 5.0]}
    eps, scores = rank_execution_providers(latencies)
    assert eps[0] == "ort-cuda"
    assert scores["ort-cuda"] < scores["ort-cpu"]


def test_fragmentation_penalty_increases_with_switches():
    nodes = ["n1", "n2", "n3", "n4", "n5"]
    c1 = compute_fragmentation_cost(nodes, ["n1", "n3", "n5"])
    c2 = compute_fragmentation_cost(nodes, ["n1", "n2", "n3"])
    assert c1 > c2


def test_shape_churn_avoids_tensorrt():
    rankings = ["tensorrt", "ort-cuda"]
    backend = select_best_backend(rankings, fragmentation_cost=2.0, shape_churn_score=0.9)
    assert backend == "ort-cuda"
