import numpy as np

def _attention(Q, K, V, scale):
    """Numerically stable scaled-dot-product attention."""
    logits = Q @ K.T * scale
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    attn_weights = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return attn_weights @ V

def _kv_error(K, V, K_hat, V_hat):
    return (np.mean((K - K_hat) ** 2) + np.mean((V - V_hat) ** 2)) / 2.0

def _oracle(Q, K, V, K_hat, V_hat, scale):
    O = _attention(Q, K, V, scale)
    O_hat = _attention(Q, K_hat, V_hat, scale)
    output_mse = float(np.mean((O - O_hat) ** 2))
    kv_err = float(_kv_error(K, V, K_hat, V_hat))
    amp = output_mse / kv_err if kv_err > 0 else 0.0
    return output_mse, kv_err, amp

def grade(sol, fx) -> dict:
    specs = [
        (8, 8, 0.001),
        (16, 8, 0.01),
        (16, 16, 0.1),
        (32, 4, 0.01),
        (32, 16, 0.001),
    ]

    total_mse = 0.0
    n_cases = len(specs)

    for seed, (d, seq_len, noise) in enumerate(specs, start=1):
        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(d)

        Q = rng.randn(seq_len, d)
        K = rng.randn(seq_len, d)
        V = rng.randn(seq_len, d)
        K_hat = K + noise * rng.randn(seq_len, d)
        V_hat = V + noise * rng.randn(seq_len, d)

        ref_mse, ref_kv, ref_amp = _oracle(Q, K, V, K_hat, V_hat, scale)

        try:
            res = sol.kv_quant_error_propagation(
                Q, K, V, K_hat, V_hat, scale=scale
            )
            lm = float(res["output_mse"])
            lk = float(res["kv_error"])
            la = float(res["amplification"])
        except Exception:
            return {"mse": 1.0}

        total_mse += (
            (lm - ref_mse) ** 2
            + (lk - ref_kv) ** 2
            + (la - ref_amp) ** 2
        ) / 3.0

    return {"mse": total_mse / n_cases}
