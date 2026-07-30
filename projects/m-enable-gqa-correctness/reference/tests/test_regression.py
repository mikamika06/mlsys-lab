import sys

import numpy as np

sys.path.insert(0, ".")
from gqa import scaled_dot_product_attention


def _qkv(B, Hkv, n_rep, L, S, D, seed):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, Hkv * n_rep, L, D))
    k = rng.standard_normal((B, Hkv, S, D))
    v = rng.standard_normal((B, Hkv, S, D))
    return q, k, v


def test_perturbing_one_kv_head_only_moves_its_own_group():
    B, Hkv, n_rep, L, S, D = 2, 3, 2, 4, 4, 5
    q, k, v = _qkv(B, Hkv, n_rep, L, S, D, seed=0)
    base = scaled_dot_product_attention(q, k, v, enable_gqa=True)
    rng = np.random.default_rng(1)
    for kv_idx in range(Hkv):
        k2, v2 = k.copy(), v.copy()
        k2[:, kv_idx] += rng.standard_normal((B, S, D))
        v2[:, kv_idx] += rng.standard_normal((B, S, D))
        out = scaled_dot_product_attention(q, k2, v2, enable_gqa=True)
        changed = ~np.all(np.isclose(base, out), axis=(0, 2, 3))
        expected = np.zeros(Hkv * n_rep, dtype=bool)
        expected[kv_idx * n_rep:(kv_idx + 1) * n_rep] = True
        assert np.array_equal(changed, expected), (
            f"perturbing kv head {kv_idx} changed heads {np.nonzero(changed)[0].tolist()}, "
            f"expected {np.nonzero(expected)[0].tolist()}"
        )


def test_enable_gqa_is_a_no_op_when_heads_match():
    B, Hkv, n_rep, L, S, D = 2, 4, 1, 5, 5, 6
    q, k, v = _qkv(B, Hkv, n_rep, L, S, D, seed=2)
    a = scaled_dot_product_attention(q, k, v, enable_gqa=True)
    b = scaled_dot_product_attention(q, k, v, enable_gqa=False)
    assert np.allclose(a, b), "enable_gqa changed the result when q_heads == kv_heads"


def test_causal_output_ignores_future_keys():
    B, Hkv, n_rep, L, D = 1, 2, 2, 4, 3
    S = L
    q, k, v = _qkv(B, Hkv, n_rep, L, S, D, seed=3)
    out_causal = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    v2 = v.copy()
    v2[:, :, -1, :] += 1000.0
    out2 = scaled_dot_product_attention(q, k, v2, is_causal=True, enable_gqa=True)
    assert np.allclose(out_causal[:, :, :-1], out2[:, :, :-1]), (
        "changing the last key/value changed earlier causal outputs"
    )
