import numpy as np
from q4k.quant import dequantize_q4_k, quantize_q4_k


def find_dominating_subblock(weights: np.ndarray) -> int:
    data = quantize_q4_k(weights)
    recon = dequantize_q4_k(data)
    sub_orig = weights.reshape(8, 32)
    sub_recon = recon.reshape(8, 32)
    mses = [np.mean((sub_orig[i] - sub_recon[i]) ** 2) for i in range(8)]
    return int(np.argmax(mses))
