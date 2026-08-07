import numpy as np
from rectatt.probe import compute_causal_mask, compute_offset
from rectatt.attention import sdpa_rectangular_causal, flash_attn_sim


def test_rectangular_causal_alignment():
    n_q, n_kv = 16, 64
    off_br = compute_offset(n_q, n_kv, "bottom_right")
    off_tl = compute_offset(n_q, n_kv, "top_left")

    assert off_br == 48
    assert off_tl == 0

    mask_br = compute_causal_mask(n_q, n_kv, "bottom_right")
    assert mask_br[0, 48] == True
    assert mask_br[0, 49] == False
    assert mask_br[15, 63] == True


def test_fa_sdpa_parity():
    np.random.seed(42)
    q = np.random.randn(2, 8, 16, 32)
    k = np.random.randn(2, 8, 64, 32)
    v = np.random.randn(2, 8, 64, 32)

    out_sdpa = sdpa_rectangular_causal(q, k, v, alignment="bottom_right")
    out_fa = flash_attn_sim(q, k, v, is_causal=True, alignment="bottom_right")

    np.testing.assert_allclose(out_sdpa, out_fa, atol=1e-5)
