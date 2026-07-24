import numpy as np

def kv_quant_error_propagation(Q, K, V, K_hat, V_hat, scale=None):
    """Compute how KV quantization error propagates through softmax attention.

    Returns dict with keys: output_mse, kv_error, amplification.
    """
    raise NotImplementedError("Implement this function")
