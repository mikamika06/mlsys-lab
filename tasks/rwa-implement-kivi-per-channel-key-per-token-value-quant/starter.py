import numpy as np


def kivi_quant_errors(K: np.ndarray, V: np.ndarray, q: np.ndarray, bits: int) -> np.ndarray:
    """
    K, V: (n_tokens, d) fp64 key/value cache. q: (d,) fp64 query.
    bits: quantizer bit-width.

    Quantize keys PER-CHANNEL (one scale/zero-point per column, computed
    across all tokens) and values PER-TOKEN (one scale/zero-point per
    row, computed across all channels) -- the KIVI scheme.

    Returns np.array([
        k_mse_per_channel,   # MSE of per-channel-quantized K vs true K
        k_mse_per_tensor,    # MSE of a per-tensor-quantized K baseline vs true K
        attn_max_abs_err,    # max abs error of attention(K,V,q) using the
                              # (per-channel K, per-token V) KIVI-quantized
                              # cache, vs the exact fp64 attention output
    ])
    """
    raise NotImplementedError('your code here')
