import numpy as np


def per_head_scale_attention_errors(
    K: np.ndarray, V: np.ndarray, q: np.ndarray, percentile: float
) -> np.ndarray:
    """
    K, V: (n, d) fp64 -- one attention head's key/value cache.
    q: (d,) fp64 query vector.
    percentile: percentile (0-100) used for the percentile-clipped scale.

    Quantize K to FP8 E4M3 with two different per-head scales -- amax
    scale (scale = max(|K|) / 448) and percentile scale (scale =
    percentile(|K|, percentile) / 448) -- and compare the resulting
    softmax(K_hat @ q / sqrt(d)) @ V attention output against the exact
    fp64 attention output.

    Returns np.array([attn_max_abs_err_amax, attn_max_abs_err_percentile]).
    """
    raise NotImplementedError('your code here')
