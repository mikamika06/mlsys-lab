import sys
import numpy as np

sys.path.insert(0, ".")
from fused_attn.causal import block_split_causal_attention_forward
from fused_attn.flops import compute_attention_flops, derive_tflops


def test_causal_masking_invariant():
    np.random.seed(42)
    B, H, N, D = 1, 2, 64, 16
    sm_scale = 1.0 / np.sqrt(D)

    Q = np.random.randn(B, H, N, D).astype(np.float64)
    K = np.random.randn(B, H, N, D).astype(np.float64)
    V = np.random.randn(B, H, N, D).astype(np.float64)

    out = block_split_causal_attention_forward(Q, K, V, sm_scale, block_size=16)

    V_modified = V.copy()
    V_modified[:, :, 32:, :] += 10.0

    out_prefix = block_split_causal_attention_forward(Q[:, :, :32, :], K[:, :, :32, :], V_modified[:, :, :32, :], sm_scale, block_size=16)

    np.testing.assert_allclose(out[:, :, :32, :], out_prefix, rtol=1e-5, atol=1e-5)


def test_flops_derivation():
    B, H, N, D = 2, 8, 1024, 64
    f_non_causal = compute_attention_flops(B, H, N, D, causal=False)
    f_causal = compute_attention_flops(B, H, N, D, causal=True)

    assert f_non_causal == 4 * B * H * N * N * D
    assert f_causal == 2 * B * H * N * N * D

    tf = derive_tflops(B, H, N, D, wall_clock_seconds=0.01, causal=False)
    expected_tf = (f_non_causal / 0.01) / 1e12
    assert abs(tf - expected_tf) < 1e-6
