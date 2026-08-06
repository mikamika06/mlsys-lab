import numpy as np
from moe.capacity import compute_expert_capacity
from moe.dispatch import grouped_gemm_dispatch, naive_expert_loop


def test_capacity_bound():
    cap = compute_expert_capacity(100, 8, 1.0, 2)
    assert cap >= 1


def test_dispatch_equivalence():
    np.random.seed(42)
    tokens = np.random.randn(32, 16).astype(np.float32)
    topk_indices = np.random.randint(0, 4, size=(32, 2))
    topk_weights = np.random.rand(32, 2).astype(np.float32)
    expert_weights = np.random.randn(4, 16, 16).astype(np.float32)
    capacity = 20

    out_grouped = grouped_gemm_dispatch(tokens, topk_indices, topk_weights, expert_weights, capacity)
    out_naive = naive_expert_loop(tokens, topk_indices, topk_weights, expert_weights, capacity)
    np.testing.assert_allclose(out_grouped, out_naive, rtol=1e-5, atol=1e-5)
