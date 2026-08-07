import sys
import numpy as np

sys.path.insert(0, ".")
from edge_export.layer_decisions import allocate_layer_strategies
from edge_export.sparsity import compare_sparsity_error
from edge_export.joint_budget import find_optimal_joint_budget


def test_layer_decisions_respect_global_budget():
    np.random.seed(42)
    weights = [np.random.randn(16, 16) for _ in range(3)]
    target_budget = 2.0
    res = allocate_layer_strategies(weights, target_budget, 2, 4, [2, 4])
    assert res is not None, "Allocation returned None"
    avg_bits = sum(r["effective_bits"] for r in res) / len(res)
    assert avg_bits <= target_budget + 1e-6, f"Average effective bits {avg_bits} exceeds target {target_budget}"


def test_sparsity_error_nm_greater_or_equal():
    np.random.seed(42)
    w = np.random.randn(16, 16)
    res = compare_sparsity_error(w, 2, 4)
    assert res["nm_mse"] >= res["unstructured_mse"] - 1e-7


def test_optimal_joint_budget_returns_valid_structure():
    np.random.seed(42)
    w = np.random.randn(16, 16)
    res = find_optimal_joint_budget(w, 2.0, [2, 4])
    assert "effective_bits" in res
    assert "mse" in res
    assert res["effective_bits"] <= 2.0 + 1e-6
