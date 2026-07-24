import numpy as np


def per_head_kv_attention(K, V, q):
    """Per-head amax E4M3FN quantize/dequantize K and V, then attend.

    K, V: float64 arrays (S, H, D) -- raw, unquantized keys/values.
    q: float64 array (H, D) -- one query vector per head.

    For each head h independently: compute scale_h = max(abs(K[:,h,:])) / 448
    (and similarly for V), quantize/dequantize K[:,h,:] and V[:,h,:] to E4M3FN
    using that head's own scale, then run scaled dot-product attention.

    Returns float64 array (H, D).
    """
    raise NotImplementedError('your code here')
