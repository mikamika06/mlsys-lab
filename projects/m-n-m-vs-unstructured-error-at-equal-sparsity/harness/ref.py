import numpy as np

np.random.seed(42)

LAYER_WEIGHTS_1 = [
    np.random.randn(16, 16),
    np.random.randn(16, 16) * 0.5,
    np.random.randn(16, 16) * 2.0,
]

LAYER_WEIGHTS_2 = [
    np.random.randn(32, 16),
    np.random.randn(32, 16) * 1.5,
]


def check_sparsity_comparison(weights, n, m):
    from edge_export.sparsity import compare_sparsity_error
    return compare_sparsity_error(weights, n, m)


def check_joint_budget(weights, max_bits, bit_options):
    from edge_export.joint_budget import find_optimal_joint_budget
    return find_optimal_joint_budget(weights, max_bits, bit_options)


def check_layer_decisions(layer_weights, target_total_bits, n, m, bit_options):
    from edge_export.layer_decisions import allocate_layer_strategies
    return allocate_layer_strategies(layer_weights, target_total_bits, n, m, bit_options)
