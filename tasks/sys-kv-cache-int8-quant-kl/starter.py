import numpy as np


def kv_cache_int8_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """Single-head scaled dot-product attention against an int8-quantized
    KV cache.

    Q : (m, d) fp32 queries.
    K, V : (n, d) fp32 keys/values, as they would be written into the KV
        cache.

    Quantize K and V to int8 with a PER-TOKEN (per-row) symmetric scale
    (scale_i = max(|row_i|) / 127, codes_i = round(row_i / scale_i)
    clipped to [-127, 127]), dequantize, then run standard scaled
    dot-product attention against the dequantized cache:

        logits  = (Q @ K_hat.T) / sqrt(d)
        weights = softmax(logits, axis=-1)
        out     = weights @ V_hat

    Returns
    -------
    logits : (m, n) array
        Pre-softmax attention scores computed from the dequantized K.
    out : (m, d) array
        The attention output computed from the dequantized K and V.
    """
    raise NotImplementedError('your code here')
