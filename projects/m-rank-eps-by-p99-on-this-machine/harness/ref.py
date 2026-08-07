import random
import numpy as np


def get_test_data():
    rng = random.Random(42)
    latencies = {
        "ort-cpu": [rng.gauss(50, 5) for _ in range(100)] + [200.0],
        "ort-cuda": [rng.gauss(10, 2) for _ in range(100)] + [25.0],
        "tensorrt": [rng.gauss(6, 1) for _ in range(100)] + [15.0]
    }
    nodes = [f"node_{i}" for i in range(10)]
    supported = [f"node_{i}" for i in range(10) if i % 2 == 0]
    return latencies, nodes, supported


def rank_execution_providers(latencies):
    scores = {}
    for ep, times in latencies.items():
        arr = np.array(times, dtype=np.float64)
        p99 = np.percentile(arr, 99)
        scores[ep] = float(p99)
    sorted_eps = sorted(scores.keys(), key=lambda x: (scores[x], x))
    return sorted_eps, scores


def compute_fragmentation_cost(subgraph_nodes, trt_supported_nodes):
    total = len(subgraph_nodes)
    if total == 0:
        return 0.0
    supported = set(trt_supported_nodes)
    is_supported = np.array([1 if n in supported else 0 for n in subgraph_nodes])
    switches = np.sum(np.abs(np.diff(is_supported)))
    fraction_unsupported = 1.0 - (np.sum(is_supported) / total)
    cost = float(switches * 2.5 + fraction_unsupported * 10.0)
    return cost


def select_best_backend(rankings, fragmentation_cost, shape_churn_score):
    best_ep = rankings[0]
    if shape_churn_score > 0.7 and best_ep == "tensorrt":
        return "ort-cuda"
    if fragmentation_cost > 15.0 and best_ep == "tensorrt":
        return "ort-cuda"
    return best_ep
