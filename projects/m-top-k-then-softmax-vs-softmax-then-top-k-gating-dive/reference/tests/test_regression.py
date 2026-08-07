import numpy as np
from moegating.dispatch import build_mixtral_dispatch_tensor
from moegating.gating import softmax_then_top_k, top_k_then_softmax


def test_gating_sum_to_one():
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((16, 8))
    w1, _ = top_k_then_softmax(logits, k=2)
    w2, _ = softmax_then_top_k(logits, k=2)
    np.testing.assert_allclose(np.sum(w1, axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.sum(w2, axis=-1), np.sum(w2, axis=-1), atol=1e-6)


def test_dispatch_tensor_indices():
    selected = np.array([[0, 2], [1, 2], [0, 1]])
    dispatch = build_mixtral_dispatch_tensor(selected, num_experts=4)
    assert dispatch.shape == (4, 3, 2)
    for t in range(3):
        for k in range(2):
            exp = selected[t, k]
            assert dispatch[exp, t, k] == 1
    assert np.sum(dispatch) == 6
