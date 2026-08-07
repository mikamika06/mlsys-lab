import sys
import numpy as np

sys.path.insert(0, ".")
from sdpa.reference import numpy_sdpa


def test_causal_mask_prevents_future_attention():
    B, H, L, D = 1, 1, 4, 8
    q = np.ones((B, H, L, D))
    k = np.random.randn(B, H, L, D)
    v = np.random.randn(B, H, L, D)

    out1 = numpy_sdpa(q, k, v, is_causal=True)

    v[0, 0, 3, :] = 999.0
    out2 = numpy_sdpa(q, k, v, is_causal=True)

    assert np.allclose(out1[:, :, 0, :], out2[:, :, 0, :]), "First query token was affected by the last KV token"
