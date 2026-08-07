import numpy as np
from kvblock.objective import (
    calculate_internal_fragmentation,
    calculate_prefix_truncation_loss,
    evaluate_workload_objective,
)


def test_internal_fragmentation_upper_bound():
    seqs = [15, 33, 64, 1, 100]
    block_size = 16
    frag = calculate_internal_fragmentation(seqs, block_size)
    assert frag < len(seqs) * block_size
    assert frag >= 0


def test_truncation_loss_non_negative():
    shared = [(1, 2, 3, 4, 5, 6, 7, 8)]
    requests = [(1, 2, 3, 4, 5, 9)]
    loss_16 = calculate_prefix_truncation_loss(shared, requests, 16)
    loss_4 = calculate_prefix_truncation_loss(shared, requests, 4)
    assert loss_16 == 5
    assert loss_4 == 1


def test_objective_cost_scaling():
    seqs = [10, 20, 30]
    shared = [(1, 2, 3, 4, 5)]
    requests = [(1, 2, 3, 4, 5)]
    cost1 = evaluate_workload_objective(seqs, shared, requests, 4, 1.0)
    cost2 = evaluate_workload_objective(seqs, shared, requests, 4, 10.0)
    assert cost2 >= cost1
