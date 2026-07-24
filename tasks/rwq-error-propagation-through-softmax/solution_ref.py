import numpy as np

def kv_quant_error_propagation(Q, K, V, K_hat, V_hat, scale=None):
    """Compute how KV quantization error propagates through softmax attention.

    Returns dict with keys: output_mse, kv_error, amplification.
    """
    d = Q.shape[-1]
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    # --- numerically stable softmax helper ---
    def _attn(Q_, K_, V_):
        logits = Q_ @ K_.T * scale
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return weights @ V_

    O = _attn(Q, K, V)
    O_hat = _attn(Q, K_hat, V_hat)

    output_mse = float(np.mean((O - O_hat) ** 2))
    kv_error = float(
        (np.mean((K - K_hat) ** 2) + np.mean((V - V_hat) ** 2)) / 2.0
    )
    amplification = output_mse / kv_error if kv_error > 0 else 0.0

    return {
        "output_mse": output_mse,
        "kv_error": kv_error,
        "amplification": amplification,
    }
