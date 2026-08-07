import numpy as np
from adapter.shape import flash_to_sdpa, sdpa_to_flash
from adapter.ref_attn import compute_lse, reference_attention


def test_shape_roundtrip():
    q = np.random.randn(2, 4, 16, 32).astype(np.float32)
    k = np.random.randn(2, 4, 16, 32).astype(np.float32)
    v = np.random.randn(2, 4, 16, 32).astype(np.float32)
    q_f, k_f, v_f = sdpa_to_flash(q, k, v)
    q_back = flash_to_sdpa(q_f)
    assert q_back.shape == q.shape
    assert np.allclose(q_back, q)


def test_lse_consistency():
    q = np.random.randn(1, 2, 8, 16).astype(np.float32)
    k = np.random.randn(1, 2, 8, 16).astype(np.float32)
    v = np.random.randn(1, 2, 8, 16).astype(np.float32)
    out, lse = reference_attention(q, k, v)
    lse_direct = compute_lse(q, k)
    assert np.allclose(lse, lse_direct, atol=1e-5)
