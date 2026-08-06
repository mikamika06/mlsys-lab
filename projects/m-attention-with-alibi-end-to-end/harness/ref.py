import numpy as np

def compute_alibi_slopes(num_heads):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(np.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if np.log2(num_heads).is_integer():
        return np.array(get_slopes_power_of_2(num_heads), dtype=np.float32)
    else:
        closest_pow2 = 2 ** int(np.floor(np.log2(num_heads)))
        slopes = get_slopes_power_of_2(closest_pow2)
        extra = get_slopes_power_of_2(2 * closest_pow2)[1::2][:num_heads - closest_pow2]
        return np.array(slopes + extra, dtype=np.float32)

def alibi_attention(query, key, value, scale=None, softcap=None):
    b, h, q_len, d = query.shape
    k_len = key.shape[2]
    if scale is None:
        scale = 1.0 / np.sqrt(d)

    slopes = compute_alibi_slopes(h)
    q_idx = np.arange(q_len)[:, None]
    k_idx = np.arange(k_len)[None, :]
    rel_pos = k_idx - q_idx
    alibi_bias = slopes[None, :, None, None] * rel_pos[None, None, :, :]

    scores = np.matmul(query, key.transpose(0, 1, 3, 2)) * scale
    scores = scores + alibi_bias

    if softcap is not None and softcap > 0:
        scores = softcap * np.tanh(scores / softcap)

    causal_mask = np.triu(np.ones((q_len, k_len), dtype=bool), k=1)
    scores[:, :, causal_mask] = -1e9

    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    out = np.matmul(probs, value)
    return out

BACKENDS = ["standard", "flash_attn", "paged_attn"]
MODIFIERS = ["alibi", "softcap", "causal", "sliding_window"]

SUPPORT_MATRIX = {
    "standard": {"alibi": True, "softcap": True, "causal": True, "sliding_window": True},
    "flash_attn": {"alibi": True, "softcap": False, "causal": True, "sliding_window": True},
    "paged_attn": {"alibi": True, "softcap": True, "causal": True, "sliding_window": False},
}

def check_support(backend, modifiers):
    if backend not in SUPPORT_MATRIX:
        return False
    return all(SUPPORT_MATRIX[backend].get(m, False) for m in modifiers)

def compute_overflow_rate(scores, threshold=65504.0, softcap=None):
    if softcap is not None and softcap > 0:
        modified_scores = softcap * np.tanh(scores / softcap)
    else:
        modified_scores = scores
    overflows = np.abs(modified_scores) > threshold
    return float(np.mean(overflows))
