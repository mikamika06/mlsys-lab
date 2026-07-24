import numpy as np


def int8_kv_attention(Q, K, V):
    """Scaled dot-product attention with per-token INT8 quantized K and V.

    Q: (n_q, d)   K, V: (n_kv, d) / (n_kv, d_v)

    Quantize K and V row-wise (one symmetric absmax int8 scale per
    key/value token: scale = max(|row|) / 127, code = clip(round(row /
    scale), -127, 127), dequant = code * scale), then run
    softmax(Q K^T / sqrt(d)) V against the dequantized K, V.

    Returns (out, mse):
      out -- float64 array, shape (n_q, d_v), attention output through the
             int8 KV path.
      mse -- float, mean squared error vs. the full-precision attention
             output computed from the un-quantized K, V.
    """
    raise NotImplementedError('your code here')
