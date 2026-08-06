import numpy as np


def compute_rope_inv_freqs(method, head_dim, original_max_len, target_max_len, base=10000.0, scale_factor=1.0, yarn_beta_fast=32.0, yarn_beta_slow=1.0):
    dims = np.arange(0, head_dim, 2, dtype=np.float64)
    base_inv_freq = 1.0 / (base ** (dims / head_dim))
    s = scale_factor if scale_factor > 1.0 else (target_max_len / original_max_len)

    if method == "linear":
        return base_inv_freq / s
    elif method == "dynamic_ntk":
        new_base = base * (s ** (head_dim / (head_dim - 2.0)))
        return 1.0 / (new_base ** (dims / head_dim))
    elif method == "yarn":
        wavelengths = 2.0 * np.pi / base_inv_freq
        low_b = original_max_len / yarn_beta_fast
        high_b = original_max_len / yarn_beta_slow

        res = np.zeros_like(base_inv_freq)
        for i, w in enumerate(wavelengths):
            if w < low_b:
                gamma = 0.0
            elif w > high_b:
                gamma = 1.0
            else:
                gamma = (w - low_b) / (high_b - low_b)
            interpolated = base_inv_freq[i] / s
            res[i] = (1.0 - gamma) * base_inv_freq[i] + gamma * interpolated
        return res
    else:
        raise ValueError(f"Unknown method {method}")


def evaluate_synthetic_perplexity(logits, targets, inv_freqs, seq_len, scale_factor=1.0, attention_factor=1.0):
    t = np.arange(seq_len, dtype=np.float64)
    dim_pairs = len(inv_freqs)

    pos_enc_cos = np.cos(np.outer(t, inv_freqs))
    pos_signal = np.sum(pos_enc_cos, axis=1, keepdims=True)

    scaled_logits = (logits + 0.1 * pos_signal) * attention_factor

    max_l = np.max(scaled_logits, axis=-1, keepdims=True)
    exp_l = np.exp(scaled_logits - max_l)
    probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

    nll = -np.log(np.take_along_axis(probs, targets[:, None], axis=-1) + 1e-12)
    mean_nll = np.mean(nll)
    return float(np.exp(mean_nll))
