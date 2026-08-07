import numpy as np
from q4k.quant import quantize_q4_k, dequantize_q4_k

def compare_q4k_q40(x):
    x = np.asarray(x, dtype=np.float32)
    b_k = quantize_q4_k(x)
    deq_k = dequantize_q4_k(b_k)
    mse_k = float(np.mean((x - deq_k) ** 2))
    mn, mx = x.min(), x.max()
    scale_0 = (mx - mn) / 15.0 if mx > mn else 1.0
    q_0 = np.clip(np.round((x - mn) / scale_0), 0, 15)
    deq_0 = q_0 * scale_0 + mn
    mse_0 = float(np.mean((x - deq_0) ** 2))
    return {"mse_q4k": mse_k, "mse_q40": mse_0}
