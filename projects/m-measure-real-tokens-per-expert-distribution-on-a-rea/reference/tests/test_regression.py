import numpy as np
import sys
sys.path.insert(0, ".")
from moe_dist.metrics import analyze_imbalance, compute_expert_load
from moe_dist.routing import measure_sparsity_pathology, simulate_loss_free_routing


def test_imbalance_detection():
    logits = np.zeros((100, 8))
    logits[:, 0] = 10.0
    counts = compute_expert_load(logits, top_k=1)
    metrics = analyze_imbalance(counts, num_experts=8)
    assert metrics["starved_experts"] == 7
    assert metrics["peak_ratio"] == 8.0
    assert metrics["cv"] > 2.0


def test_sparsity_pathology_trend():
    np.random.seed(42)
    layers = [np.random.randn(200, 16) for _ in range(4)]
    res = measure_sparsity_pathology(layers, [1, 4])
    assert res[1]["cv"] > res[4]["cv"]
    assert res[1]["peak_ratio"] >= res[4]["peak_ratio"]
