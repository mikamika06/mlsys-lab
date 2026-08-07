import numpy as np
from moe.dispatch import combine_tokens, dispatch_tokens


def test_capacity_limit_enforced():
    tokens = np.random.randn(10, 16)
    indices = np.zeros((10, 1), dtype=np.int64)
    weights = np.ones((10, 1), dtype=np.float32)
    num_experts = 2
    capacity = 3

    buffers, meta = dispatch_tokens(tokens, indices, weights, num_experts, capacity)
    assert len(meta["routes"]) == 3
    assert np.count_nonzero(buffers[0]) <= 3 * 16


def test_reconstruction_identity():
    tokens = np.random.randn(8, 16)
    indices = np.array([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=np.int64)
    weights = np.ones((8, 1), dtype=np.float32)
    num_experts = 2
    capacity = 10

    buffers, meta = dispatch_tokens(tokens, indices, weights, num_experts, capacity)
    out = combine_tokens(buffers, meta)
    np.testing.assert_allclose(tokens, out, atol=1e-5)
